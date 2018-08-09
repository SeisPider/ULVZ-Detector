# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose: Modules
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 23:06h, 03/08/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
#-------------------------------------------------------------------------------
"""Modules to handle objects or methods of back-projection technology
"""
from obspy.signal.filter import envelope
import numpy as np
from .distaz import DistAz
from . import logger
from .utils import obtain_travel_time
from obspy.taup import TauPyModel
import multiprocessing as mp 

class BackProjector(object):
    """Object to project the amplitude back to the source area
    """
    def __init__(self, stream, model="PREM"):
        """initialization

        Parameters
        ==========
        stream: Obspy.core.Stream
            Waveform to be projected, it should be a collection of 
            traces read from SAC files
        model: str
            Specific 1D earth's reference model
        """
        self.model = TauPyModel(model=model)
        self.st = stream

    def back_projection(self, marker, phase, mesh, depth=0.0, table_grid=0.01, 
                        **kwargs):
        """Project the amplitude to the source area

        Parameters
        ==========
        marker: str
            Marker the time of the main phase
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        mesh: src.mesh.Mesh2DArea obj.
            specified source region
        """
        # # Obtain distances
        # self.dists = self._dist_table(mesh)

        # # Obtain time table
        # self.distrange, self.arrivals = self._time_table(table_grid, depth, phase)

        # Append source location to obj.
        self.slon, self.slat, self.sdp = self._check_source()
        
        # Obtain time of the repeat source to each station
        self.source_receiver_time = self._source_receiver_time_table(phase)

        # Compute the travel time from the source to all possible scatter locations
        tempresult = self._source_scatter_time_table(mesh, phase, depth)
        self.source_scatter_gcarcs, self.source_scatter_times = tempresult

        # Compute the travel time from all possible scatter locations to each receivers
        tempresult = self._scatter_receiver_time_table(mesh, phase, depth)
        self.scatter_receiver_gcarcs, self.scatter_receiver_time = tempresult

        # Project each trace to each source patch
        amppatchs = []
        for idx, item in enumerate(self.st):
            onepatch = self._project_onetr(item, mesh, marker, phase, idx, **kwargs)
            amppatchs.append(onepatch)
        amppatchs = np.array(amppatchs)
        return amppatchs 

    def _check_source(self):
        """Check the consistence and obtain the location of source from seismic SAC traces
        """
        logger.info("Checking source location ......")
        # Check consistence of source location
        hds = lambda x: np.array([tr.stats.sac[x] for tr in self.st])
        lons, lats, dps = hds('evlo'), hds('evla'), hds('evdp')
        npred = lambda y: (y[:-2]==y[1:-1]).all()
        if ~(npred(lons) and npred(lats) and npred(dps)):
            raise ValueError("Repeat source location inconsistence !")
        
        # Get the source location
        slon, slat, sdp = lons[0], lats[0], dps[0]
        msg = "Suc. obtain source location"
        logger.info(msg + " ({:.5f},{:.5f},{:.5f})!".format(slon, slat, sdp))

        return slon, slat, sdp
 

    def _scatter_receiver_time_table(self, mesh, phase, depth):
        """Compute travel time from the possible receiver location to 
        each receivers

        Parameters
        ==========
        mesh: src.mesh.Mesh2DArea obj.
            specified source region
        depth: float
            the depth of the assumed scatter, in km
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        """
        logger.info("Calculating distance table ......")
        # estimate the distances
        dists, times = [], []
        latlat, lonlon = mesh.latlat, mesh.lonlon
        for trace in self.st:
            # Get header
            sachd = trace.stats.sac
            
            # Get the receiver location
            reclon, reclat = sachd["stlo"],  sachd["stla"]

            # Compute source-receiver distances
            gcarcs = np.array([DistAz(latlat[i][j], lonlon[i][j], 
                                      reclat, reclon).getDelta() 
                                      for i in range(mesh.shape[0])
                                      for j in range(mesh.shape[1])])
            rdp = 0.0 # Assume all stations locate at earth's surface
            times1d = np.array([obtain_travel_time(self.model, depth, rdp, x, phase)
                                for x in gcarcs])
            dists.append(gcarcs.reshape(mesh.shape))
            times.append(times1d.reshape(mesh.shape))
        logger.info("Suc. Calculate distance table from scatter to receivers !")

        return dists, times

    def _source_receiver_time_table(self, phase):
        """Compute travel time from the repeat source to traces

        Parameters
        ==========
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        """
        logger.info("Calculating travel time from repeat source to stations ......")
        
        # Obtain source location
        slon, slat, sdp = self.slon, self.slat, self.sdp

        # Compute travel time from repeat source to each station
        
        distrange = np.zeros(len(self.st))
        for idx, trace in enumerate(self.st):
            # Get header
            sachd = trace.stats.sac
            # Get the receiver location
            reclon, reclat = sachd["stlo"],  sachd["stla"]
            distrange[idx] = np.array([DistAz(slat, slon, reclat, reclon).getDelta()])
        
        rdp = 0.0 # Assume all stations locate at earth's surface
        arrivals = np.array([obtain_travel_time(self.model, sdp, rdp, x, phase) for x in distrange])
        logger.info("Suc. calculate travel time from repeat source to stations ......")
        return arrivals

    def _source_scatter_time_table(self, mesh, phase, depth):
        """Compute travel time from source to each possible scatter location

        Parameter
        =========
        phase: list of str
            TauP phase
        mesh: src.mesh.Mesh2DArea obj.
            specified possible scatter region
        depth: float
            Specified scatter depth, in km
        """
        # Retrive location
        latlat, lonlon = mesh.latlat, mesh.lonlon

        # Obtain the source-scatter distances
        gcarcs1d = np.array([DistAz(latlat[i][j], lonlon[i][j], 
                                  self.slat, self.slon).getDelta() 
                                  for i in range(mesh.shape[0])
                                  for j in range(mesh.shape[1])])

        # Compute the travel time from each source-scatter pair
        times1d = np.array([obtain_travel_time(self.model, self.sdp, depth, x, phase)
                            for x in gcarcs1d])
        return gcarcs1d.reshape(mesh.shape), times1d.reshape(mesh.shape)


    def _project_onetr(self, trace, mesh, marker, phase, 
                       tridx, toenv=True, norm=True):
        """same as the func. name

        Parameters
        ==========
        toenv: Bool
            Determine wheather thansfer the waveform to envelope
        norm: Bool
            Determine wheather normalize the waveform
        mesh: src.mesh.Mesh2DArea obj.
            specified possible scatter region
        source_region: src.mesh.Mesh2DArea obj.
            specified source region
        """
        # Get header
        sachd = trace.stats.sac
        
        # Check the iztype
        if sachd["iztype"] != 11:
            raise ValueError("Reference time of SAC file is not event source time")
        timescale = np.arange(sachd['b'], sachd['e'], sachd['delta'])

        # Obtain the reference time from markered time
        reft, srctrtime = sachd[marker], self.source_receiver_time[tridx]

        # Time shift between the picked arrival and predicted arrival
        shift = reft - srctrtime

        # Check the amplitude at each time
        def check_amp(timept, timescale, amp):
            """Check the amplitude at particular time
            """
            if timept < timescale.min() or timept > timescale.max():
                return np.nan
            idx = (np.abs(timescale - timept)).argmin()
            return amp[idx]

        # Obtain envelope 
        if toenv:
            data = envelope(trace.data)
        else:
            data = np.abs(trace.data)
        
        # Norm waveform
        if norm:
            data /= data.max()

        # Interpolate to obtain the travel time from each patch to 
        # this receiver
        time_table = self.scatter_receiver_time
        amppatchs =  np.array([check_amp(time_table[tridx][latidx][lonidx]+shift, timescale, data) 
                               for latidx in range(mesh.shape[0])
                               for lonidx in range(mesh.shape[1])])
        return amppatchs

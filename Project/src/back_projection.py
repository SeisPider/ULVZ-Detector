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
from obspy.taup import TauPyModel
from scipy import interpolate
import matplotlib.pyplot as plt

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
        self.model = model
        self.st = stream

    def back_projection(self, marker, phase, source_region, depth=0.0, table_grid=0.01, 
                        timewind=(-20, 20), **kwargs):
        """Project the amplitude to the source area

        Parameters
        ==========
        marker: str
            Marker the time of the main phase
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        source_region: src.mesh.Mesh2DArea obj.
            specified source region
        timewind: tuple
            time window for projection
        """
        # Obtain distances
        self.dists = self._dist_table(source_region)

        # Obtain time table
        self.distrange, self.arrivals = self._time_table(table_grid, depth, phase)

        # Project each trace to each source patch
        amppatchs = []
        for idx, item in enumerate(self.st):
            onepatch = self._project_onetr(item, marker, phase, source_region, idx, **kwargs)
            amppatchs.append(onepatch)
        amppatchs = np.array(amppatchs)

        # Investigate the stacked possible source regions
        return amppatchs 


    def _dist_table(self, source_region):
        logger.info("Calculating distance table ......")
        # estimate the distances
        dists = []
        latlat, lonlon = source_region.latlat, source_region.lonlon
        for trace in self.st:
            # Get header
            sachd = trace.stats.sac
            
            # Get the receiver location
            reclon, reclat = sachd["stlo"],  sachd["stla"]

            # Compute source-receiver distances

            gcarcs = np.array([DistAz(latlat[i][j], lonlon[i][j], 
                                      reclat, reclon).getDelta() 
                                      for i in range(latlat.shape[0])
                                      for j in range(latlat.shape[1])])
            dists.append(gcarcs)
        logger.info("Suc. Calculate distance table !")
        return dists
    
    def _time_table(self, grid, depth, phase):
        """Estimate the times at each grid points

        Parameters
        ==========
        grid: float
           grid in computing the travel time
        depth: float
            the depth of assumed source, in km
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        """
        logger.info("Calculating travel time table ......")
        # distance range
        dists = np.array(self.dists).flatten()
        distrange = np.arange(dists.min(), dists.max()+2*grid, grid)

        # Compute travel time table
        model = TauPyModel(model=self.model)
        arrivals = np.array([model.get_travel_times(source_depth_in_km=depth,
                             distance_in_degree=x, phase_list=phase)[0].time
                             for x in distrange])
        logger.info("Suc. Calculate travel time table !")
        return distrange, arrivals


    def _project_onetr(self, trace, marker, phase, source_region, 
                       tridx, toenv=True, norm=True):
        """same as the func. name

        Parameters
        ==========
        toenv: Bool
            Determine wheather thansfer the waveform to envelope
        norm: Bool
            Determine wheather normalize the waveform

        source_region: src.mesh.Mesh2DArea obj.
            specified source region
        """
        # Get header
        sachd = trace.stats.sac
        
        # Check the iztype
        if sachd["iztype"] != 11:
            raise ValueError("Reference time of SAC file is not event source time")
        timescale = np.arange(sachd['b'], sachd['e'], sachd['delta'])

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
        dists = self.dists[tridx]
        f = interpolate.interp1d(self.distrange, self.arrivals)
        timepatchs = f(dists)

        amppatchs =  np.array([check_amp(x, timescale, data) for x in timepatchs])
        return amppatchs

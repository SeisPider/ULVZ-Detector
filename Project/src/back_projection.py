# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose: Modules
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 23:06h, 03/08/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
"""Modules to handle objects or methods of back-projection technology

References
==========
1.  Hedlin, Michael AH, Peter M. Shearer, and Paul S. Earle. "Seismic evidence 
    for small-scale heterogeneity throughout the Earth's mantle." 
    Nature 387.6629 (1997): 145.
2.  Wen, Lianxing. "Intense seismic scattering near the Earth's core‐mantle 
    boundary beneath the Comoros hotspot." Geophysical research letters 
    27.22 (2000): 3627-3630.
3.  Yao, Jiayuan, and Lianxing Wen. "Seismic structure and ultra-low velocity 
    zones at the base of the Earth’s mantle beneath Southeast Asia." 
    Physics of the Earth and Planetary Interiors 233 (2014): 103-111.
"""
from obspy.signal.filter import envelope
import numpy as np
from .distaz import DistAz
from . import logger
from .utils import obtain_travel_time, isolate
from obspy.taup import TauPyModel
from scipy import interpolate


class BackProjector(object):
    """Object to project the amplitude back to the source area
    """

    def __init__(self, stream, model="prem"):
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

    def back_projection(self, marker, phases, mesh, depth=0.0,
                        table_grid=0.01, **kwargs):
        """Project the amplitude to the source area

        Parameters
        ==========
        marker: str
            Marker the time of the main phase
        phases: tuple of str.
            Specific the phases used in this computatiopn, 
            it should be distinguishable for TauP. It includes 
            three phases (source_scatter_phase, scatter_receiver_phase, 
            source_receiver_phase), e.g. (P, PKIKP, PKIKP).
        mesh: src.mesh.Mesh2DArea obj.
            specified source region
        """
        # Obtain phases
        source_scatter_phase, scatter_receiver_phase, source_receiver_phase = phases

        # Append source location to obj.
        self.slon, self.slat, self.sdp = self._check_source()

        # Obtain time of the repeat source to each traces
        tempresult = self._source_receiver_time_table(source_receiver_phase)
        self.source_receiver_gcarcs, self.source_receiver_time = tempresult

        # Compute the travel time from the source to all possible scatter locations
        tempresult = self._source_scatter_time_table(
            mesh, source_scatter_phase, depth)
        self.source_scatter_gcarcs, self.source_scatter_times = tempresult

        # Compute the travel time from all possible scatter locations to each receivers
        tempresult = self._scatter_receiver_time_table(
            mesh, scatter_receiver_phase, depth, table_grid)
        self.scatter_receiver_gcarcs, self.scatter_receiver_time = tempresult

        # Project each trace to each source patch
        amppatchs, hits = self._project_traces(mesh, marker, **kwargs)
        return np.array(amppatchs), np.array(hits)

    def _check_source(self):
        """Check the consistence and obtain the location of source from seismic SAC traces
        """
        logger.info("Checking source location ......")
        # Check consistence of source location

        def hds(x): return np.array([tr.stats.sac[x] for tr in self.st])
        lons, lats, dps = hds('evlo'), hds('evla'), hds('evdp')

        def npred(y): return (y[:-2] == y[1:-1]).all()
        if ~(npred(lons) and npred(lats) and npred(dps)):
            raise ValueError("Repeat source location inconsistence !")

        # Get the source location
        slon, slat, sdp = lons[0], lats[0], dps[0]
        msg = "Suc. obtain source location"
        logger.info(msg + " ({:.5f},{:.5f},{:.5f})!".format(slon, slat, sdp))

        return slon, slat, sdp

    def _scatter_receiver_time_table(self, mesh, phase, depth, distgrid=0.1):
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
        distgrid: float
            Distance grid to compute travel time for later interpolating, in degree 
        """
        logger.info("Calculating distance table ......")
        # estimate the distances
        dists = []
        latlat, lonlon = mesh.latlat, mesh.lonlon
        for trace in self.st:
            sachd = trace.stats.sac
            # Get the receiver location
            reclon, reclat = sachd["stlo"],  sachd["stla"]

            # Compute scatter-receiver distances
            gcarcs = np.array([DistAz(latlat[i][j], lonlon[i][j],
                                      reclat, reclon).getDelta()
                               for i in range(mesh.shape[0])
                               for j in range(mesh.shape[1])])
            dists.append(gcarcs.reshape(mesh.shape))

        # Calculate travel time from minimum source-scatter distance to the maximum one
        gcarcs = np.array(dists).flatten()
        mindist, maxdist = gcarcs.min() - 2*distgrid, gcarcs.max() + 2*distgrid
        refdists = np.arange(mindist, maxdist, distgrid)
        rdp = 0.0  # Assume all stations locate at earth's surface
        reftimes = np.array([obtain_travel_time(self.model, depth, rdp, x, phase)
                             for x in refdists])

        # Interpolate to obtain travel time from each possible scatter location to receivers
        f = interpolate.interp1d(refdists, reftimes)
        times = []
        for tridx in range(len(dists)):
            diststr = dists[tridx]
            timestr = np.array([f(diststr[i][j]) for i in range(mesh.shape[0])
                                for j in range(mesh.shape[1])])
            times.append(timestr.reshape(mesh.shape))
        logger.info("Suc. Calculate distance table from scatter to receivers !")
        return dists, times

    def _source_receiver_time_table(self, phase):
        """Compute travel time from the repeat source to traces

        Parameters
        ==========
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        """
        logger.info(
            "Calculating travel time from repeat source to stations ......")

        # Obtain source location
        slon, slat, sdp = self.slon, self.slat, self.sdp

        # Compute travel time from repeat source to each station

        distrange = np.zeros(len(self.st))
        for idx, trace in enumerate(self.st):
            # Get header
            sachd = trace.stats.sac
            # Get the receiver location
            reclon, reclat = sachd["stlo"],  sachd["stla"]
            distrange[idx] = np.array(
                [DistAz(slat, slon, reclat, reclon).getDelta()])

        rdp = 0.0  # Assume all stations locate at earth's surface
        arrivals = np.array(
            [obtain_travel_time(self.model, sdp, rdp, x, phase) for x in distrange])
        logger.info(
            "Suc. calculate travel time from repeat source to stations !")
        return distrange, arrivals

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
    # def scatter_energe_computation(self,)

    def _project_traces(self, mesh, marker, norm=True, ahead=1,
                        wind_pkikp=(-1, 2)):
        """same as the func. name

        Parameters
        ==========
        norm: Bool
            Determine wheather normalize the waveform
        mesh: src.mesh.Mesh2DArea obj.
            specified possible scatter region
        trace: ObsPy.Trace
            waveform for projecting
        marker: str
            marker of PKIKP arrival time
        tridx: int
            index of this trace in waveform Stream
        ahead: float
            Only waveforms at least this time ahead PKIKP marker are thought to be 
            precursor
        wind_pkikp: tuple
            Time window encloses the PKIKP phase which is relative time refering marker 
            time
        ref_gcarc: None or float
            The reference epicentral distance where the scatter energy will be corrected to  
        """
        # ################################################################################
        # Obtain the scatter energe
        # ################################################################################

        scatter_energe = []
        # Compute the scatter energe , cut out the waveforms of precursors
        for idx, item in enumerate(self.st):
            # Get header
            sachd = item.stats.sac

            # Obtain the left and right boundary of precursor in time
            time_table = (self.source_scatter_times + self.scatter_receiver_time[idx]).flatten()
        

            # Check the iztype
            if sachd["iztype"] != 11:
                raise ValueError(
                    "Reference time of SAC file is not event source time")
            timescale = np.arange(sachd['b'], sachd['e'], sachd['delta'])

            # Obtain the reference time from markered time
            reft, srctrtime = sachd[marker], self.source_receiver_time[idx]

            # Time table by considering shift between theoretical and the real arrival time
            shift = reft - srctrtime

            time_table +=  shift
            pre_begin, pre_end = time_table.min(), time_table.max()-ahead
            
            # pre_end = min(pre_end, reft)
            print(pre_begin, pre_end, reft)

            # Obtain envelope
            data = envelope(item.data)

            # Norm waveform
            if norm:
                # Take the maximum amplitude in enclosed window to be that of PKIKP
                lftbound, rightbound = reft+wind_pkikp[0], reft+wind_pkikp[1]
                msk = isolate(timescale, lftbound, rightbound)
                data /= data[msk].max()

            # Isolate and store the precursor part
            msk = isolate(timescale, pre_begin, pre_end)
            subdict = {"time": timescale, "energe": data, "shift": shift}
            scatter_energe.append(subdict)

        # ################################################################################
        # Epicentral distance correction
        # ################################################################################
        # Note: Hedlin et al,1997(ref. 1) found the precursor energe rises corresponding
        # to the epicentral distance increasing. Thus, for energe staking, we
        # should correct them to specific epicentral distance

        # Compute the maximum scatter energe
        max_energe = np.array([x["energe"].max() for x in scatter_energe])

        # Obtain the trace index with the median epicentral distance
        dists = self.source_receiver_gcarcs
        median_idx = np.argsort(dists)[len(dists)//2]
        self.ref_gcarc = dists[median_idx]

        # correct all scatter energe to the trace with median epicentral
        # distance
        factor = max_energe[median_idx]/max_energe
        for idx, item in enumerate(scatter_energe):
            item["energe"] *= factor[idx]

        # ################################################################################
        # Back-projection
        # ################################################################################

        # Definition of function
        def check_amp(timept, timescale, amp):
            """Check the amplitude at particular time
            """
            if timept < timescale.min() or timept > timescale.max():
                return np.nan
            idx = (np.abs(timescale - timept)).argmin()
            return amp[idx]

        # Project energe back

        # Calculate time from source to receivers via scatter
        time_table = [self.source_scatter_times +
                      x for x in self.scatter_receiver_time]
        # Assign energy to each possible scatter locations
        amppatchs, hits = [], np.zeros(mesh.shape)

        # Loop over traces
        for idx, item in enumerate(scatter_energe):
            timescale, energe, shift = item['time'], item['energe'], item['shift']
            amppatch = np.zeros(mesh.shape)

            # loop over possible scatter locations
            for latidx in range(mesh.shape[0]):
                for lonidx in range(mesh.shape[1]):

                    # Check the amplitude
                    extracted_amp = check_amp(
                        time_table[idx][latidx][lonidx]+shift, timescale, energe)
                    if ~np.isnan(extracted_amp):
                        hits[latidx][lonidx] += 1
                    amppatch[latidx][lonidx] = extracted_amp
            amppatchs.append(amppatch)
        return amppatchs, hits

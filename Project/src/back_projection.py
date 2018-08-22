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
from .utils import obtain_travel_time, isolate, precursor_correction
import matplotlib.pyplot as plt
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

    def back_projection(self, marker, phases, mesh, depth=0.0, table_grid=0.1,
                        **kwargs):
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
        depth: float
            Assumed scatter depth
        table_grid: float
           The minimum geographical grid in computing travel time from 
           scatters to receivers
        """
        # Obtain phases
        source_scatter_phase, scatter_receiver_phase, source_receiver_phase = phases

        # Append source location to obj.
        self.slon, self.slat, self.sdp = self._check_source()

        # Obtain time of the source to each traces
        tempresult = self._source_receiver_time_table(source_receiver_phase)
        self.source_receiver_gcarcs, self.source_receiver_time = tempresult

        # Compute the travel time from the source to all possible scatter locations
        tempresult = self._source_scatter_time_table(
            mesh, source_scatter_phase, depth)
        self.source_scatter_gcarcs, self.source_scatter_times = tempresult

        # Compute the travel time from all possible scatter locations to each receivers
        tempresult = self._scatter_receiver_time_table(
            mesh, scatter_receiver_phase, depth, distgrid=table_grid)
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
            raise ValueError("Source location inconsistence !")

        # Get the source location
        slon, slat, sdp = lons[0], lats[0], dps[0]
        msg = "Suc. obtain source location"
        logger.info(msg + " ({:.5f},{:.5f},{:.5f})!".format(slon, slat, sdp))

        return slon, slat, sdp

    def _scatter_receiver_time_table(self, mesh, phase, depth, distgrid):
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
        logger.info(
            "Calculating travel times from scatters to receivers ......")
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
        logger.info(
            "Suc. calculate travel times from scatters to receivers !")
        return dists, times

    def _source_receiver_time_table(self, phase):
        """Compute travel time from the source to traces

        Parameters
        ==========
        phase: list of str.
            Specific the main phase, it used be distinguishable for TauP
        """
        logger.info(
            "Calculating travel time from source to receivers ......")

        # Obtain source location
        slon, slat, sdp = self.slon, self.slat, self.sdp

        # Compute travel time from source to each station

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
            "Suc. calculate travel times from source to receivers !")
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
        logger.info(
            "Calculating travel times from source to scatters ......")
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
        logger.info(
            "Suc. calculate travel times from source to scatters !")
        return gcarcs1d.reshape(mesh.shape), times1d.reshape(mesh.shape)

    def _project_traces(self, mesh, marker, min_factor=0.5, norm=True, env=False,
                        correction=False, ahead=1, wind_main_phase=(-1, 2), debug=False):
        """same as the func. name

        Parameters
        ==========
        norm: Bool
            Determine wheather normalize the waveform
        env: Bool
           Determine to project envelope of the waveforms or not
        correction: Bool
           Determine to correct precursor amplitude with hedlin's work,ref. 1
        mesh: src.mesh.Mesh2DArea obj.
            specified possible scatter region
        marker: str
            marker of the main arrival time
        ahead: float
            Only waveforms at least this time ahead PKIKP marker are thought to be 
            precursor
        wind_main_phase: tuple
            Time window encloses the main phase which is relative time refering to marker 
            time, needed when using it to compute the scatter energe
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
            time_table = (self.source_scatter_times +
                          self.scatter_receiver_time[idx]).flatten()

            # Construct time coordinate and obtain the reference time from
            # markered phase arrival
            reft, srctrtime = sachd[marker], self.source_receiver_time[idx]
            timescale = np.arange(sachd['b'], sachd['e'], sachd['delta'])

            # Time table by considering shift between theoretical and
            # the real arrival time
            shift = reft - srctrtime
            time_table += shift
            pre_begin, pre_end = np.nanmin(time_table), np.nanmax(time_table)

            # Apart main phase
            pre_end = min(pre_end, reft)
            # Ignore precursor phases several seconds ahead from the main arrival
            # Which may be mispicking of main phase energe
            pre_end -= ahead
            if debug:
                # Compute travel time difference between scatter reflected wave and direct PKIKP
                times = self.source_scatter_times + \
                    self.scatter_receiver_time[idx]
                times -= srctrtime

                # Display time difference
                plt.contourf(mesh.latlat, mesh.lonlon, times,
                             cmap=plt.get_cmap('seismic'),
                             extend='both', alpha=0.5)
                plt.colorbar()
                plt.show()

                # Give out numerical results
                print("Source-Scatter" + "*"*66)
                print(self.source_scatter_times)
                print("Scatter-Receiver" + "*"*64)
                print(self.scatter_receiver_time[idx])
                print(time_table)

            # Obtain envelope
            if env:
                data = envelope(item.data)
            else:
                data = item.data

            # Norm waveform
            if norm:
                # Take the maximum amplitude in enclosed window to be that of the main phase
                lftbound, rightbound = reft + \
                    wind_main_phase[0], reft+wind_main_phase[1]
                msk = isolate(timescale, lftbound, rightbound)
                data /= data[msk].max()

            # Isolate and store the precursor part
            msk = isolate(timescale, pre_begin, pre_end)
            subdict = {"time": timescale[msk],
                       "energy": data[msk], "shift": shift}
            scatter_energe.append(subdict)

        # ################################################################################
        # Epicentral distance correction
        # ################################################################################
        # Note: Hedlin et al,1997(ref. 1) found the precursor energe rises corresponding
        # to the epicentral distance increasing. Thus, for energe staking, we
        # should correct them to specific epicentral distance

        # Use precursor amplitude correction from hedline's paper
        if correction:
            factor = precursor_correction(self.source_receiver_gcarcs)
            for idx, item in enumerate(scatter_energe):
                item["energy"] /= factor[idx]

        # ################################################################################
        # Back-projection
        # ################################################################################

        # Definition of function
        def check_amp(timept, timescale, amp):
            """Check the amplitude at particular time
            """
            # Time range constrains
            if timept < timescale.min() or timept > timescale.max():
                return np.nan
            # Time reliability constrains
            if np.isnan(timept):
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
            timescale, energe, shift = item['time'], item['energy'], item['shift']
            amppatch = np.zeros(mesh.shape)

            # loop over possible scatter locations
            for latidx in range(mesh.shape[0]):
                for lonidx in range(mesh.shape[1]):

                    # Check and retrive the amplitudes
                    extracted_amp = check_amp(
                        time_table[idx][latidx][lonidx]+shift, timescale, energe)
                    if ~np.isnan(extracted_amp):
                        hits[latidx][lonidx] += 1
                    amppatch[latidx][lonidx] = extracted_amp
            amppatchs.append(amppatch)

            # Debug part
            if debug:
                print(amppatchs)
        return amppatchs, hits

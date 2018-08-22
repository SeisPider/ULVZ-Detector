# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose: Store project related methods
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 20:02h, 07/08/2018
#        Usage: Modulus
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy
from easyprocess import EasyProcess
from scipy import interpolate


def obtain_travel_time(taupmodel, sdp, rdp, gcarc, pha):
    """same as func. name
    """

    try:
        # Check units
        if sdp > 7000:
            sdp /= 1000  # Tranfer it to km.
        if rdp > 7000:
            rdp /= 1000  # Tranfer it to km.

        # If the scatter is deeper than the source, revert the station
        # and source characteristics
        if sdp < rdp:
            temp = deepcopy(rdp)
            rdp = deepcopy(sdp)
            sdp = deepcopy(temp)
        
        # Compute the first arrival
        p = EasyProcess('taup_time -mod "{}" -deg "{}" -h "{}" --stadepth "{}" --ph "{}" --time'.format(
            taupmodel, gcarc, sdp, rdp, pha)).call()
        # p = EasyProcess('taup_time -mod "{}" -deg "{}" -h "{}" --stadepth "{}"  --time'.format(
        #     "prem", gcarc, sdp, rdp)).call()
        time = [float(x) for x in p.stdout.strip().split()][0]
        return time
    except IndexError:
        return np.nan


def isolate(scale, left_bound, right_bound):
    """isolate the scale ruler based on given left and right bounds

    Parameters
    ==========
    scale: numpy.array
        Scale ruler
    left_bound: float
        The left boundary
    right_bound: float
        The right boundary
    """
    condition = (scale >= left_bound) * (scale <= right_bound)
    return np.where(condition)

def precursor_correction(dists):
    """Calculate the prevursor amplitude correction

    Parameters
    ==========
    dists: float
        gcarc to obtain precursor amplitude correction  
    """
    # This observation were measured from Hedline,1997 with ruler
    gcarcs = np.arange(131, 142)
    factors = np.array([0.2, 0.3, 0.3, 0.5, 0.4, 0.5, 0.6, 0.8, 0.8, 1.1, 1.4])
    precursor_corrector = interpolate.interp1d(gcarcs, factors, kind="cubic",
                                               fill_value="extrapolate")
    return precursor_corrector(dists)

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
            "prem", gcarc, sdp, rdp, pha)).call()
        # p = EasyProcess('taup_time -mod "{}" -deg "{}" -h "{}" --stadepth "{}"  --time'.format(
        #     "prem", gcarc, sdp, rdp)).call()
        time = [float(x) for x in p.stdout.strip().split()][0]
        # print(rdp, sdp, gcarc, pha)
        return time
    except IndexError:
        # print(rdp, sdp, gcarc, pha)
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

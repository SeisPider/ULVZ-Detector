# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose: Store project related methods
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 20:02h, 07/08/2018
#        Usage: Modulus
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
#-------------------------------------------------------------------------------
import numpy as np
def obtain_travel_time(taupmodel, sdp, rdp, gcarc, pha):
    """same as func. name
    """
    try:
        # If the scatter is deeper than the source, revert the station
        # and source characteristics
        if sdp > rdp:
            temp = sdp
            rdp = sdp
            sdp = temp
        return taupmodel.get_travel_times(source_depth_in_km=sdp,
                                          receiver_depth_in_km=rdp,
                                          distance_in_degree=gcarc, 
                                          phase_list=pha)[0].time
    except IndexError:
        return np.nan
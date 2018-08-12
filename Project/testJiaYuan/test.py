# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 20:00h, 05/08/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
#-------------------------------------------------------------------------------
import sys
sys.path.append("../")
from src.back_projection import BackProjector
from src.mesh import Mesh2DArea
import numpy as np
import matplotlib.pyplot as plt
from obspy import read

if __name__ == '__main__':
    mesh = Mesh2DArea(-20, 20, 90, 128)
    mesh.standard_mesh(grid=4)
    eventid = "20110310170836"
    st = read("../process/{}/TA.L*BHZ".format(eventid))
    bp = BackProjector(st)
    marker, phase, grid ="t1", ("p", "PKIKP", "PKIKP"), 0.5 
    amppatchs, hits = bp.back_projection(marker, phase, mesh, depth=2890.0, 
                                   table_grid=grid, norm=True)
    aveamp = np.nanmean(amppatchs, axis=0)
    np.savetxt("{}.energe.csv".format(eventid), aveamp, fmt="%.5f")
    np.savetxt("{}.hits.csv".format(eventid), hits, fmt="%.5f")
    np.savetxt("{}.coord.lon.csv".format(eventid), mesh.lonlon, fmt="%.5f")
    np.savetxt("{}.coord.lat.csv".format(eventid), mesh.latlat, fmt="%.5f")
    
    plt.contourf(mesh.latlat, mesh.lonlon, aveamp, 
                 cmap=plt.get_cmap('seismic'),
                 extend='both', alpha=0.5)
    plt.colorbar()
    plt.show()

    plt.contourf(mesh.latlat, mesh.lonlon, hits, 
                 cmap=plt.get_cmap('seismic'),
                 extend='both', alpha=0.5)
    plt.colorbar()
    plt.show()

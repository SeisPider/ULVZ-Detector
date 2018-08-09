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
    mesh = Mesh2DArea(37, 53, 71, 100)
    mesh.standard_mesh(grid=0.1)
    st = read("../../Data/LanLan/20141120031926_20150126044310/*.CC.SAC")
    bp = BackProjector(st)
    marker, phase, grid ="t2", ["S"], 0.1
    amppatchs = bp.back_projection(marker, phase, mesh, depth=0.0, 
                                   table_grid=grid, toenv=False, 
                                   norm=False)
    shape = mesh.shape
    aveamp = np.nanmean(amppatchs, axis=0)
    plt.contourf(mesh.latlat, mesh.lonlon, aveamp.reshape(shape), 
                 cmap=plt.get_cmap('seismic'),
                 extend='both', alpha=0.5)
    plt.colorbar()
    plt.show()

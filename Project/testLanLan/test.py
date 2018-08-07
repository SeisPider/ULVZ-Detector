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
    mesh = Mesh2DArea(39, 50, 75, 95)
    mesh.standard_mesh(grid=0.1)
    st = read("../../Data/LanLan/20141120031926_20150126044310/2014*.t.SAC")
    bp = BackProjector(st)
    marker, phase, grid ="t1", ["P"], 0.1
    amppatchs = bp.back_projection(marker, phase, mesh, depth=13, 
                                   table_grid=grid, toenv=True, norm=True)
    
    aveamp = np.zeros(mesh.shape)
    for item in amppatchs:
        aveamp += item
    aveamp /= len(amppatchs)
    
    shape = mesh.lonlon.shape
    plt.contourf(mesh.latlat, mesh.lonlon, aveamp.reshape(shape), 
                 cmap=plt.get_cmap('seismic'),
                 extend='both', alpha=0.5)
    plt.colorbar()
    plt.show()

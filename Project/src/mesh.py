# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose: Modulus
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 22:14h, 29/06/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
import numpy as np
from . import logger

class Mesh2DArea(object):
    """construct mesh of a 2D source region
    """
    def __init__(self, minlat, maxlat, minlon, maxlon):
        """initial setup of 2D region
        """
        self.latbox = (minlat, maxlat)
        self.lonbox = (minlon, maxlon)
    
    def standard_mesh(self, grid, latlon2xy=True):
        """mesh research area with standard format

        Parameters
        ==========
        latgrid: float
            grid in meshing latitude direction
        grid: float
            grid in meshing
        latlon2xy: bool
            determine whether translate from geographyical coordinate 
            to cartesian coordinate 
        """
        # LOG info.
        logger.info("Meshing area covering {} in latitude and {} in longitude".format(self.latbox, self.lonbox))

        # set the location info. for each grid
        self.grid = grid
        lats = np.arange(self.latbox[0], self.latbox[1], grid)
        lons = np.arange(self.lonbox[0], self.lonbox[1], grid)
        self.latlat, self.lonlon = np.meshgrid(lats, lons)
        # set shape
        self.shape = self.latlat.shape

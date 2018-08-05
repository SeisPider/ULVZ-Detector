# -*coding: utf-8 -*-
"""
ULVZDetector
========

A software for detecting the ULVZ from enumerous earthquakes
"""
import doctest
import logging

DEG2KM = 111.190 # km
EARTH_R = 6371 # km

# Setup the logger
FORMAT = "[%(asctime)s]  %(levelname)s: %(message)s"
logging.basicConfig(
    #filename="logger.info",
    level=logging.INFO,
    format=FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


__title__ = "ULVZDetector"
__version__ = "0.0.1"
__author__ = "Xiao Xiao"
__license__ = "MIT"
__copyright__ = "Copyright 2018-2018 Xiao Xiao"



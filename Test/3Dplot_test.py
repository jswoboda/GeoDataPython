# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 14:25:27 2015

@author: anna
"""

from GeoData import GeoData
from GeoData import utilityfuncs
from pylab import *
import numpy as np
from GeoData.plotting import *


risrName = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'
risr_class = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
risr_class.changedata('nel','ne',revpower,[10.0])

geodata = risr_class
altlist = [300.0, 400.0, 450.0]
xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
vbounds = [5e10,5e11]
title='Ne altitude slices at 300, 400 and 500 km'
plot3D(geodata, altlist, xyvecs, vbounds, title)
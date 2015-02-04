# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 10:35:52 2015

@author: anna
"""
from GeoData import GeoData
from GeoData import utilityfuncs
from pylab import *
import numpy as np
from GeoData.plotting import *


risrName = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'
omtiName = '/Users/anna/Research/Ionosphere/Semeter/OMTIdata.h5'
omti_class = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
risr_class = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
risr_class.changedata('nel','ne',revpower,[10.0])

geodatalist = [omti_class, risr_class]
altlist = [300]
xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
vbounds = [[200,800],[5e10,5e11]]
title='OMTI data and NEL linear interpolation'
alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title)


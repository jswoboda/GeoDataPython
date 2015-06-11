# -*- coding: utf-8 -*-
"""
Example code on using the alt_slice_overlay function in the plotting.py file.
Takes two h5 files--a RISR file and an OMTI file-- and creates 2 objects. This is inputed into alt_slice_overlay.
The output is a 2D colorplot with the OMTI data on the bottom in grayscale and the RISR parameter on top with an alpha of 0.4


@author: Anna Stuhlmacher
"""
from __future__ import division, absolute_import
import numpy as np
#
debug=True
if debug: #use code without installing
    import sys;  sys.path.append('../')

from GeoData import GeoData
from GeoData import utilityfuncs

def revpower(x1,x2):
    return x2**x1

#path names to h5 files
risrName = 'ran120219.004.hdf5'
omtiName = 'OMTIdata.h5'

#creating GeoData objects of the 2 files, given a specific parameter
omti = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
risr = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
#converting logarthmic electron density (nel) array into electron density (ne) array
risr.changedata('nel','ne',revpower,[10])

#first object in geodatalist is being overlayed over by the second object
geodatalist = [omti, risr]
altlist = [300]
xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
vbounds = [[200,800],[5e10,5e11]]
title='OMTI data and NE linear interpolation'

#%% This requires Mayavi, currently only for Python2
import GeoData.plotting as GP
GP.alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title)


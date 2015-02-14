# -*- coding: utf-8 -*-
"""
INCOMPLETE...plots blank graph underneath ax2 handle

@author: Anna Stuhlmacher
"""
import matplotlib.pyplot as plt
from GeoData import GeoData
from GeoData import utilityfuncs
from pylab import *
import numpy as np
from GeoData.plotting import *

def revpower(x1,x2):
    return np.power(x2,x1)

#path names to h5 files
risrName = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'
omtiName = '/Users/anna/Research/Ionosphere/Semeter/OMTIdata.h5'

#creating GeoData objects of the 2 files, given a specific parameter
omti_class = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
risr_class = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
#converting logarthmic electron density (nel) array into electron density (ne) array
risr_class.changedata('nel','ne',revpower,[10.0])

#first object in geodatalist is being overlayed over by the second object
geodatalist_slice = [omti_class, risr_class]
geodatalist_contour = [omti_class.copy(), risr_class.copy()]
altlist = [300]
xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
vbounds = [[200,800],[5e10,5e11]]
title='OMTI data and NE linear interpolation'

fig3, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5), facecolor='white')
ax1 = fig3.add_subplot(121)
ax1 = alt_slice_overlay(geodatalist_slice, altlist, xyvecs, vbounds, title, axis=ax1)
ax2 = fig3.add_subplot(122)
ax2 = alt_contour_overlay(geodatalist_contour, altlist, xyvecs, vbounds, title, axis=ax2)

ax1.set_ylabel('y')
ax1.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlabel('x')

# Show only fig3
fig3.show()

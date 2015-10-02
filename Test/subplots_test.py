#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code on using the alt_slice_overlay function in the plotting.py file.
Takes two h5 files--a RISR file and an OMTI file-- and creates 2 objects. This is inputed into alt_slice_overlay.
The output is a 2D colorplot with the OMTI data on the bottom in grayscale and the RISR parameter on top with an alpha of 0.4
@author: Anna Stuhlmacher, Michael Hirsch


first you need to install GeoDataPython by python setup.py develop
"""
from __future__ import division, absolute_import
from matplotlib.pyplot import subplots,show
import numpy as np
#
import GeoData.plotting as GP
#
from load_isropt import load_risromti

def plotisropt(risrName,omtiName):

    risr,omti = load_risromti(risrName,omtiName)
    #first object in geodatalist is being overlayed over by the second object
    altlist = [300]
    xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = [[200,800],[5e10,5e11]]
    title='OMTI data and NE linear interpolation'

    fig3, (ax1, ax2) = subplots(1,2,figsize=(10,5), facecolor='white')
    ax1 = fig3.add_subplot(121)
    ax1 = GP.alt_slice_overlay((omti, risr), altlist, xyvecs, vbounds, title, axis=ax1)
    ax2 = fig3.add_subplot(122)
    ax2 = GP.alt_contour_overlay((omti, risr), altlist, xyvecs, vbounds, title, axis=ax2)

    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')

if __name__ == '__main__':
    plotisropt(risrName='~/data/ran120219.004.hdf5',
               omtiName='~/data/OMTIdata.h5')
    show()

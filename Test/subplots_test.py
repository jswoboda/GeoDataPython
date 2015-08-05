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
from GeoData import GeoData
from GeoData import utilityfuncs
import GeoData.plotting as GP

def revpower(x1,x2):
    return x2**x1

def plotisropt(risrName,omtiName):

    omti,risr = load_testdata(risrName,omtiName)
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

def load_testdata(risrName,omtiName):
    #creating GeoData objects of the 2 files, given a specific parameter
    omti = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
    risr = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
    #converting logarthmic electron density (nel) array into electron density (ne) array
    risr.changedata('nel','ne',revpower,[10.0])

    return omti,risr

if __name__ == '__main__':
    plotisropt(risrName='ran120219.004.hdf5',omtiName='OMTIdata.h5')
    show()

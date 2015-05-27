# -*- coding: utf-8 -*-
"""
Example code on using the plot3D function in the plotting.py file.
Takes an h5 RISR file and creates a GeoData object. This is inputed into plot3D.
The output is 3D image with different colorplot slices at different altitudes.

@author: Anna Stuhlmacher
"""

from GeoData import GeoData
from GeoData import utilityfuncs
from pylab import *
import numpy as np
from GeoData.plotting import *

def revpower(x1,x2):
    return np.power(x2,x1)

#path names to RISR file
risrName = 'ran120219.004.hdf5'
#creating a GeoData object of the RISR h5 file, given a specific parameter
risr_class = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))

#converting logarthmic electron density (nel) array into electron density (ne) array
risr_class.changedata('nel','ne',revpower,[10.0])

#geodata is the RISR object in which altitude slices will be taken from
geodata = risr_class
#list of altitudes where slices of the RISR data will be taken
altlist = [300.0, 400.0, 450.0]
xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
#minimum and maximum values to be plotted on colorplot
vbounds = [5e10,5e11]
title='Ne altitude slices at 300, 400 and 500 km'

#plotting.py function
plot3D(geodata, altlist, xyvecs, vbounds, title)
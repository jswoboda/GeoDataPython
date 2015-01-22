# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 09:58:20 2014

@author: anna

taking an hdf5
-saving the parameters
-transferring spherical coord -> cartesian coord
-take slice at 300km
-interpolate in geodata class (linear)
-flatten to array
-plot altitude slice (of NEL)
"""
from GeoData import utilityfuncs
#from GeoData import plotting
import numpy as np
from GeoData import GeoData_0
import scipy as sp
import matplotlib.pyplot as plt
#import pdb

def interpLin(dataClass, new_coords, interpMeth):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpDataA = gd2.data['nel']
    interpDataB = gd2.data['ti']
    pa = interpDataA[:,0].reshape(x.shape)
    pb = interpDataB[:,0].reshape(x.shape)
    return pa, pb

xvec = np.linspace(-100.0,500.0)
yvec = np.linspace(0.0,600.0)
x,y = sp.meshgrid(xvec,yvec)
z = np.ones(x.shape)*300.0
np.ndarray.flatten(x)
np.ndarray.flatten(y)
np.ndarray.flatten(z)
new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]
h5name = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'

gdL = GeoData_0.GeoData_0(utilityfuncs.readMad_hdf5,(h5name, ['ti','nel']) )
gdN = GeoData_0.GeoData_0(utilityfuncs.readMad_hdf5,(h5name, ['ti','nel']) )
   
p1a, p1b = interpLin(gdL, new_coords, 'linear')
p2a, p2b = interpLin(gdN, new_coords, 'nearest')

#plotting.overlayPlots(p2a, p1a, extent, 0.9)


f, axarr = plt.subplots(2, 2, facecolor='white')
axarr[0, 0].imshow(p1a,extent=extent,origin = 'lower')
axarr[0, 0].set_title('NEL-Linear')
axarr[0, 1].imshow(p1b,extent=extent,origin = 'lower')
axarr[0, 1].set_title('TI-Linear')
axarr[1, 0].imshow(p2a,extent=extent,origin = 'lower')
axarr[1, 0].set_title('NEL-Nearest')
axarr[1, 1].imshow(p2b,extent=extent,origin = 'lower')
axarr[1, 1].set_title('TI-Nearest')

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.suptitle('Altitude Slice at 300km')





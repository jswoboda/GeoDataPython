# -*- coding: utf-8 -*-
"""
Creates a GeoData object from an omti h5 files and outputs a 2D colorplot in gray scale.

@author: Anna Stuhlmacher

"""
from GeoData import utilityfuncs
import numpy as np
from GeoData import GeoData
import scipy as sp
import matplotlib.pyplot as plt

xvec = np.linspace(-100.0,500.0)
yvec = np.linspace(0.0,600.0)
x,y = sp.meshgrid(xvec,yvec)
z = np.ones(x.shape)*300.0
np.ndarray.flatten(x)
np.ndarray.flatten(y)
np.ndarray.flatten(z)
new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]
h5name = '/Users/anna/Research/Ionosphere/Semeter/OMTIdata.h5'

#omti takes data input from file directory
omti = GeoData.GeoData(utilityfuncs.readOMTI,(h5name, ['optical']) )

def interp(dataClass, new_coords, interpMeth):
    """
    Uses GeoData's interpolate function to smooth data,
    method used on the parameter is either 'linear' or 'nearest'.
    Returns a new interpolated array.
    """
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpData = gd2.data['optical']
    p = interpData[:,0].reshape(x.shape)
    return p
    
p = interp(omti, new_coords, 'nearest')

#Plotting in gray scale, with colorbar
fig, ax = plt.subplots(facecolor='white')
omtiplot = ax.imshow(p,origin = 'lower', aspect = 'auto',extent=extent,vmin=200,vmax=800,cmap=plt.get_cmap('gray'))
plt.title("OMTI data")
cbar=plt.colorbar(omtiplot)
cbar.set_label('OMTI')
plt.show(False)


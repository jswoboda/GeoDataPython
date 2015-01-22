# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:03:16 2014

@author: anna
"""
from GeoData import utilityfuncs
import numpy as np
from GeoData import GeoData
import scipy as sp
import matplotlib.pyplot as plt
import pdb

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

omti = GeoData.GeoData(utilityfuncs.readOMTI,(h5name, ['optical']) )

def interp(dataClass, new_coords, interpMeth):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpData = gd2.data['optical']
    p = interpData[:,0].reshape(x.shape)
    return p

#pdb.set_trace()   
p = interp(omti, new_coords, 'linear')

fig, ax = plt.subplots(facecolor='white')
omtiplot = ax.imshow(p,origin = 'lower', aspect = 'auto')
plt.title("OMTI data")
cbar=plt.colorbar(omtiplot)
cbar.set_label('OMTI')
plt.show()


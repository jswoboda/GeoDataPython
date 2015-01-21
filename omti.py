# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 22:03:16 2014

@author: anna
"""
from GeoData import utilityfuncs
from GeoData import plotting
from GeoData import CoordTransforms
import numpy as np
from GeoData import GeoData
import pdb
import scipy as sp
import scipy.interpolate as spinterp
import matplotlib.pyplot as plt

def readOMTI(filename):
    outlist = GeoData.read_h5_main(filename)
    optical = outlist[0]['optical']
    instance = optical[:,0]
    enu = outlist[2].T
    cartCoords = CoordTransforms.enu2cartisian(enu)
    coordnames = 'Cartesian'
    return (instance, coordnames, np.array(cartCoords, dtype='f'), outlist[3], np.asarray(outlist[4],dtype='f'))


h5name = '/Users/anna/Research/Ionosphere/Semeter/OMTIdata.h5'

xvec = np.linspace(-100.0,500.0) 
yvec = np.linspace(0.0,600.0) 
x,y = sp.meshgrid(xvec,yvec)
z = np.ones(x.shape)*300.0
np.ndarray.flatten(x)
np.ndarray.flatten(y)
np.ndarray.flatten(z)
new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]
outlist = GeoData.read_h5_main(h5name)
optical = outlist[0]['optical']
enu = outlist[2]
dataloc = CoordTransforms.enu2cartisian(enu)
coordnames = 'Cartesian'
sensorloc = outlist[3]
times = outlist[4]

#gd = GeoData.GeoData(utilityfuncs.readOMTI(h5name))
loclist = [1,2] ###???
times = times[loclist]
data = optical[:,loclist]

Nt = times.shape[0]
NNlocs = new_coords.shape[0]

# Loop through parameters and create temp variable

New_param = np.zeros((NNlocs,Nt),dtype=data.dtype)
for itime in np.arange(Nt):
    curparam =data[:,itime]
    pdb.set_trace()
    intparam = spinterp.griddata(dataloc,curparam,new_coords,method='linear',fill_value=np.nan)
    New_param[:,itime] = intparam
data = New_param

#PLOTTING
array_omti = data[:,0].reshape(x.shape)

fig, ax = plt.subplots(facecolor='white')
omtiplot = ax.imshow(array_omti,extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()],origin = 'lower', aspect = 'auto')
cbar=plt.colorbar(omtiplot)
cbar.set_label('OMTI)')
plt.title('Altitude Slice of OMTI at 300km \nLinear Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show(False)


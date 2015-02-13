# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 09:58:20 2014

@author: Anna Stuhlmacher

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
from GeoData import GeoData
import scipy as sp
import matplotlib.pyplot as plt
import pickle

#function used for change data
def revpower(x1,x2):
    return np.power(x2,x1)

def interpLin(dataClass, new_coords, interpMeth):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpDataA = gd2.data['ne']
    interpDataB = gd2.data['ti']
    pa = interpDataA[:,0].reshape(x.shape)
    pb = interpDataB[:,0].reshape(x.shape)
    return pa, pb

# set up interpolation
xvec = np.linspace(-100.0,500.0)
yvec = np.linspace(0.0,600.0)
x,y = sp.meshgrid(xvec,yvec)
z = np.ones(x.shape)*300.0
np.ndarray.flatten(x)
np.ndarray.flatten(y)
np.ndarray.flatten(z)
# min and max of plots
Tminmax = [500,2000]
Neminmax =np.array([5e10,5e11])

new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]
h5name = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'
#h5name = 'ran120219.004.hdf5'
gdL = GeoData.GeoData(utilityfuncs.readMad_hdf5,(h5name, ['ti','nel']) )
# change data call
gdL.changedata('nel','ne',revpower,[10.0])
gdN = gdL.copy()#GeoData.GeoData(utilityfuncs.readMad_hdf5,(h5name, ['ti','nel']) )

# a is nel
p1a, p1b = interpLin(gdL.copy(), new_coords, 'linear')
p2a, p2b = interpLin(gdN.copy(), new_coords, 'nearest')

pickle.dump((p1a, p2a), open("nel.p", "wb"))
pickle.dump((p1b, p2b), open("ti.p", "wb"))

#plotting.overlayPlots(p2a, p1a, extent, 0.9)

f, axarr = plt.subplots(2, 2, facecolor='white')
im1 = axarr[0, 0].imshow(p1a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
axarr[0, 0].set_title('NE-Linear')
cbar1 = plt.colorbar(im1,ax=axarr[0, 0])
cbar1.ax.get_yaxis().labelpad = 15
cbar1.ax.set_ylabel('$m^{-3}$', rotation=270)

im2 = axarr[0, 1].imshow(p1b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
axarr[0, 1].set_title('TI-Linear')
cbar2 = plt.colorbar(im2,ax=axarr[0, 1])
cbar2.ax.get_yaxis().labelpad = 15
cbar2.ax.set_ylabel('K', rotation=270)

im3 = axarr[1, 0].imshow(p2a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
axarr[1, 0].set_title('NE-Nearest')
cbar3 = plt.colorbar(im3,ax=axarr[1, 0])
cbar3.ax.get_yaxis().labelpad = 15
cbar3.ax.set_ylabel('$m^{-3}$', rotation=270)

im4 = axarr[1, 1].imshow(p2b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
axarr[1, 1].set_title('TI-Nearest')
cbar4 = plt.colorbar(im4,ax=axarr[1, 1])
cbar4.ax.get_yaxis().labelpad = 15
cbar4.ax.set_ylabel('K', rotation=270)

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.suptitle('Altitude Slice at 300km')
plt.show(False)





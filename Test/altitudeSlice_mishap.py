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
from __future__ import division,absolute_import
import logging
#from GeoData import plotting
import numpy as np
from matplotlib.pyplot import show,subplots
import h5py
#
from GeoData import GeoData
from GeoData import utilityfuncs

timeind = [13,14] #arbitrary user time slices

#function used for change data
def revpower(x1,x2):
    return x2**x1

def interpLin(dataClass, new_coords, interpMeth,x):
    gd2 = dataClass.timeslice(timeind)
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpDataA = gd2.data['ne']
    interpDataB = gd2.data['ti']
    pa = interpDataA[:,0].reshape(x.shape)
    pb = interpDataB[:,0].reshape(x.shape)
    return pa, pb

def demo(h5name,slicealt_km=140.0,h5out=None):
    # set up interpolation
    xvec = np.linspace(-100,500)
    yvec = np.linspace(0,600)
    x,y = np.meshgrid(xvec,yvec)
    z = np.ones_like(x)*slicealt_km

    # min and max of plots
    Tminmax = [500,2000]
    Neminmax =[1e9,5e11]

    new_coords = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    gdL = GeoData.GeoData(utilityfuncs.readMad_hdf5,(h5name, ['ti','nel']) )
    # change data call
    gdL.changedata('nel','ne',revpower,[10.0])

    # a is nel
    p1a, p1b = interpLin(gdL.copy(), new_coords, 'linear',x)
    p2a, p2b = interpLin(gdL.copy(), new_coords, 'nearest',x)

    if h5out is not None:
        print('writing '+str(h5out))
        with h5py.File('altitudeSliceOutput.h5','w',libver='latest') as fo:
            fo['/nel/p1a'] = p1a
            fo['/nel/p2a'] = p2a
            fo['/ti/p1b'] = p1b
            fo['/ti/p2b'] = p2b

    #plotting.overlayPlots(p2a, p1a, extent, 0.9)

    fg, axarr = subplots(2, 2, facecolor='white',sharex=True,sharey=True)

    if np.isnan(p1a).all(): logging.warning('p1a all nan')
    im1 = axarr[0, 0].imshow(p1a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
    plotlbl(fg,axarr[0,0],im1,'NE-Linear','$m^{-3}$')

    if np.isnan(p1b).all(): logging.warning('p1b all nan')
    im2 = axarr[0, 1].imshow(p1b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
    plotlbl(fg,axarr[0,1],im2,'TI-Linear','K')

    if np.isnan(p2a).all(): logging.warning('p2a all nan')
    im3 = axarr[1, 0].imshow(p2a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
    plotlbl(fg,axarr[1,0],im3,'NE-Nearest Neighbor','$m^{-3}$')

    if np.isnan(p2b).all(): logging.warning('p2a all nan')
    im4 = axarr[1, 1].imshow(p2b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
    plotlbl(fg,axarr[1,1],im4,'TI-Nearest Neighbor','K')

    fg.suptitle('Altitude Slice at {} km'.format(slicealt_km))

def plotlbl(fg,ax,im,txt,lbl):
    ax.set_title(txt)
    cbar1 = fg.colorbar(im,ax=ax)
    cbar1.ax.get_yaxis().labelpad = 15
    cbar1.ax.set_ylabel(lbl, rotation=270)

if __name__ == '__main__':
    demo('.hdf5',140.)
    show()





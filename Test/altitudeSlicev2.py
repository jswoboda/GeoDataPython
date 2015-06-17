#!/usr/bin/env python2
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
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py
#
try:
    from .GeoData import utilityfuncs
    from .GeoData import GeoData
except:#debug for running without install
    import sys; sys.path.append('..')
    from GeoData import utilityfuncs
    from GeoData import GeoData

#function used for change data
def revpower(x1,x2):
    return x2**x1

def interpLin(dataClass, new_coords, interpMeth,x):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpDataA = gd2.data['ne']
    interpDataB = gd2.data['ti']
    pa = interpDataA[:,0].reshape(x.shape)
    pb = interpDataB[:,0].reshape(x.shape)
    return pa, pb

def demo(h5name,h5out=None):
    # set up interpolation
    xvec = np.linspace(-100,500)
    yvec = np.linspace(0,600)
    x,y = sp.meshgrid(xvec,yvec)
    z = np.ones_like(x)*300.0

    # min and max of plots
    Tminmax = [500,2000]
    Neminmax =[5e10,5e11]

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

    fg, axarr = plt.subplots(2, 2, facecolor='white',sharex=True,sharey=True)
    im1 = axarr[0, 0].imshow(p1a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
    plotlbl(fg,axarr[0,0],im1,'NE-Linear','$m^{-3}$')

    im2 = axarr[0, 1].imshow(p1b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
    plotlbl(fg,axarr[0,1],im2,'TI-Linear','K')

    im3 = axarr[1, 0].imshow(p2a,extent=extent,origin = 'lower',vmin=Neminmax[0],vmax=Neminmax[1])
    plotlbl(fg,axarr[1,0],im3,'NE-Nearest Neighbor','$m^{-3}$')


    im4 = axarr[1, 1].imshow(p2b,extent=extent,origin = 'lower',vmin=Tminmax[0],vmax=Tminmax[1])
    plotlbl(fg,axarr[1,1],im4,'TI-Nearest Neighbor','K')

    fg.suptitle('Altitude Slice at 300km')

def plotlbl(fg,ax,im,txt,lbl):
    ax.set_title(txt)
    cbar1 = fg.colorbar(im,ax=ax)
    cbar1.ax.get_yaxis().labelpad = 15
    cbar1.ax.set_ylabel(lbl, rotation=270)

if __name__ == '__main__':
    demo('ran120219.004.hdf5','altitudeSliceTestOutput.h5')
    plt.show()





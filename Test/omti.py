# -*- coding: utf-8 -*-
"""
Creates a GeoData object from an omti h5 files and outputs a 2D colorplot in gray scale.

@author: Anna Stuhlmacher

"""
from __future__ import division,absolute_import
import numpy as np
import matplotlib.pyplot as plt
#
debug=True
if debug: #use code without installing
    import sys;  sys.path.append('../')
from GeoData import utilityfuncs
from GeoData import GeoData

def selftest(h5name):
    xvec = np.linspace(-100.,500.)
    yvec = np.linspace(0.,600.)
    x,y = np.meshgrid(xvec,yvec)
    z = np.ones_like(x)*300

    new_coords = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    extent=(xvec.min(),xvec.max(),yvec.min(),yvec.max())

    #omti takes data input from file directory
    omti = GeoData.GeoData(utilityfuncs.readOMTI,(h5name, ['optical']) )

    p = interp(omti, new_coords, x.shape,'nearest')

    #Plotting in gray scale, with colorbar
    fig = plt.figure(facecolor='white'); ax= fig.gca()
    omtiplot = ax.imshow(p,origin = 'lower', aspect = 'auto',extent=extent,
                         vmin=200,vmax=800,cmap=plt.get_cmap('gray'))
    ax.set_title("OMTI data")
    cbar=fig.colorbar(omtiplot)
    cbar.set_label('OMTI')


def interp(dataClass, new_coords, newshape,interpMeth):
    """
    Uses GeoData's interpolate function to smooth data,
    method used on the parameter is either 'linear' or 'nearest'.
    Returns a new interpolated array.
    """
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpData = gd2.data['optical']
    p = interpData[:,0].reshape(newshape)
    return p

if __name__ == '__main__':
    selftest('OMTIdata.h5')
    plt.show()

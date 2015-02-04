# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: anna

plotting
"""

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def interpRISR(dataClass, new_coords, interpMeth):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpDataA = gd2.data['ne']
    interpDataB = gd2.data['ti']
    pa = interpDataA[:,0].reshape(x.shape)
    pb = interpDataB[:,0].reshape(x.shape)
    return pa, pb
    
def interp(dataClass, x, new_coords, interpMeth, key):
    gd2 = dataClass.timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method=interpMeth, fill_value=np.nan)
    interpData = gd2.data[key]
    p = interpData[:,0].reshape(x.shape)
    return p    

def revpower(x1,x2):
    return np.power(x2,x1)
        
def alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title):
    """
    geodatalist - A list of geodata objects that will be overlayed, first object is on the bottom and in gray scale
    altlist - A list of the altitudes that we can overlay.
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for each geodata object. ie, vbounds=[[500,2000], [5e10,5e11]]
    title - A string that holds for the overall image
    Returns an image of an overlayed plot at a specific altitude.
    """
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    x,y = sp.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    np.ndarray.flatten(x)
    np.ndarray.flatten(y)
    np.ndarray.flatten(z)
    new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()] 
    
    key0 = geodatalist[0].data.keys()
    key1 =  geodatalist[1].data.keys()
    risr = interp(geodatalist[1], x, new_coords, 'linear', key1[0]) 
    omti = interp(geodatalist[0], x, new_coords, 'nearest', key0[0])
     
    plt.figure(facecolor='white')        
    bottom = imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
    cbar1 = plt.colorbar(bottom)
    cbar1.set_label(key0[0])
    hold(True)
    top = imshow(risr, cmap=cm.jet, alpha=0.4, extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
    cbar2 = plt.colorbar(top)
    cbar2.set_label(key1[0])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    show()
    



#if __name__ =='__main__':
#    risrName = '/Users/anna/Research/Ionosphere/Semeter/ran120219.004.hdf5'
#    omtiName = '/Users/anna/Research/Ionosphere/Semeter/OMTIdata.h5'
#    omti_class = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
#    risr_class = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
#    risr_class.changedata('nel','ne',revpower,[10.0])
#    
#    geodatalist = [omti_class, risr_class]
#    altlist = [300]
#    xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
#    vbounds = [[200,800],[5e10,5e11]]
#    title='OMTI data and NEL linear interpolation'
#    alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title)
#    
#    

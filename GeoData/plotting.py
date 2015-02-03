# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: anna

plotting
"""

from pylab import *
import matplotlib.pyplot as plt
import pickle

def overlayPlots (P1, P2, extent, alpha2, titlename, units):
    """Plots the 2D images of two arrays overlayed on each other. 
    P1 and P2 are arrays, where P2 is on top and P1 is in gray scale.
    Extent is the list of x and y ranges, [x min, x max, y min, y max]
    alpha2 is the transparency value of the P2, 0 < alpha2 < 1
    """
    plt.figure(facecolor='white')        
    bottom = imshow(P1, cmap=cm.gray, extent=extent, origin='lower')
    cbar1 = plt.colorbar(bottom)
    cbar1.set_label('OMTI')
    hold(True)
    top = imshow(P2, cmap=cm.jet, alpha=alpha2, extent=extent, origin='lower')
    cbar2 = plt.colorbar(top)
    cbar2.set_label(units)
    plt.title(titlename)
    plt.xlabel('x')
    plt.ylabel('y')
    show()
    
if __name__ =='__main__':
    omti = pickle.load(open("/Users/anna/Research/Ionosphere/Semeter/GeoDataPython/Test/omti.p", "rb"))
    ti_linear, ti_nearest = pickle.load(open("/Users/anna/Research/Ionosphere/Semeter/GeoDataPython/Test/ti.p", "rb"))
    nel_linear, nel_nearest = pickle.load(open("/Users/anna/Research/Ionosphere/Semeter/GeoDataPython/Test/nel.p", "rb"))
    xvec = np.linspace(-100.0,500.0)
    yvec = np.linspace(0.0,600.0)
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]
    titled = "Omti Data and TI linear interpolation"
    units = (r'log( $m^-3$ )', 'K') #nel, ti
    overlayPlots(omti, ti_linear, extent, 0.4, titled, units[1])
    
    

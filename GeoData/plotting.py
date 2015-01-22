# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: anna

plotting
"""

from pylab import *
import matplotlib.pyplot as plt

def overlayPlots (P1, P2, extent, alpha2):
    """Plots the 2D images of two arrays overlayed on each other. 
    P1 and P2 are arrays, where P2 is on top and P1 is in gray scale.
    Extent is the list of x and y ranges, [x min, x max, y min, y max]
    alpha2 is the transparency value of the P2, 0 < alpha2 < 1
    """
    plt.figure(facecolor='white')        
    imshow(P1, cmap=cm.gray, extent=extent, origin='lower')
    hold(True)
    imshow(P2, cmap=cm.jet, alpha=alpha2, extent=extent, origin='lower')
    show()
    

    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: anna

plotting
"""

from pylab import *
import matplotlib.pyplot as plt

def overlayPlots (P1, P2, extent, alpha2):
    "P1 and P2 are arrays"
    fig = plt.figure(facecolor='white')        
    im1 = imshow(P1, cmap=cm.gray, extent=extent, origin='lower')
    hold(True)
    im2 = imshow(P2, cmap=cm.jet, alpha=alpha2, extent=extent, origin='lower')
    show()
    
def implot(array1,extent, ax):
    p = ax.imshow(array1,extent=extent,origin = 'lower', aspect = 'auto')
    return p
    
    
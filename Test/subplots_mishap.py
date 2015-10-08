#!/usr/bin/env python3
"""
Example code on using the alt_slice_overlay function in the plotting.py file.
Takes two h5 files--a PFISR file and a Neo sCMOS file-- and creates 2 objects. This is inputed into alt_slice_overlay.
The output is a 2D colorplot with the optical data on the bottom in grayscale and the ISR parameter on top with an alpha of 0.4
@author:  Michael Hirsch

first you need to install GeoDataPython by python setup.py develop
"""
from __future__ import division, absolute_import
from matplotlib.pyplot import subplots,show
import numpy as np
#
import GeoData.plotting as GP
#
from load_isropt import load_pfisr_neo
#
picktimeind = [14,15] #arbitrary user time index choice

def plotisropt(isrName,optName,azelfn,heightkm):

    isr,opt = load_pfisr_neo(isrName,optName,azelfn,heightkm)
    #first object in geodatalist is being overlayed over by the second object
    altlist = [300]
    xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = [(None,None),(None,None)]
    title='Neo data and Ne linear interpolation'

    fig3, (ax1, ax2) = subplots(1,2,figsize=(10,5), facecolor='white')

    ax1 = GP.alt_slice_overlay((opt, isr), altlist, xyvecs, vbounds, title, axis=ax1,
                               picktimeind=picktimeind)

    ax2 = GP.alt_contour_overlay((opt, isr), altlist, xyvecs, vbounds, title, axis=ax2,
                                 picktimeind=picktimeind)

    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='March 1,2011 example at PFISR with Neo sCMOS camera')
    p.add_argument('--radaronly',help='load only radar data',action='store_true')
    p = p.parse_args()

    if p.radaronly:
        optName = None
        azelfn = None
    else:
        optName = '~/data/2011-03-01/110301_1043.h5'
        azelfn = '~/data/CMOS/calMishap2011Mar.h5'

    plotisropt(isrName='~/data/2011-03-01/pfa110301.003.hdf5',
               optName=optName,
               azelfn=azelfn,
               heightkm=140.)
    show()

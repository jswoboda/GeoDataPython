#!/usr/bin/env python3
"""
Data may be obtained from
https://drive.google.com/open?id=0B7P8Xeeyo_YIVTlfMk9wY0YtbzQ

Example code on using the alt_slice_overlay function in the plotting.py file.
Takes two h5 files--a PFISR file and a Neo sCMOS file-- and creates 2 objects,
which are input into alt_slice_overlay.
The output is a 2D colorplot with the optical data on the bottom in grayscale
and the ISR parameter on top with an alpha of 0.4

"""
from __future__ import division, absolute_import
from matplotlib.pyplot import subplots,show
import numpy as np
#
import GeoData.plotting as GP
#
from load_isropt import load_pfisr_neo
#
treq = [1.29905394E9,1.299053940142857E9] # unix time

def plotisropt(isrName,optName,azelfn,heightkm):

    isr,opt = load_pfisr_neo(isrName,optName,azelfn,heightkm,treq=treq)
    #first object in geodatalist is being overlayed over by the second object
    altlist = [300]
    xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = [(None,None),(None,None)]
    title='Neo data and Ne linear interpolation'

    fig3, (ax1, ax2) = subplots(1,2,figsize=(10,5))

    ax1 = GP.alt_slice_overlay((opt, isr), altlist, xyvecs, vbounds, title, ax1,treq)

    ax2 = GP.alt_contour_overlay((opt, isr), altlist, xyvecs, vbounds, title, ax2,treq)

    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='March 2011 example at PFISR with Neo sCMOS camera')
    p.add_argument('--radaronly',help='load only radar data',action='store_true')
    p = p.parse_args()

    if p.radaronly:
        optName = None
        azelfn = None
    else:
        optName = '~/data/2011-03-02/110302_0819.h5'
        azelfn = '~/data/2011-03/calMishap2011Mar.h5'

    plotisropt(isrName='~/data/2011-03-02/pfa110302.002.hdf5',
               optName=optName,
               azelfn=azelfn,
               heightkm=140.)
    show()

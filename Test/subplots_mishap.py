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
from load_isropt import load_pfisr_hst


def plotisropt(isr,opt,slicealt,tind):

    """ first object in geodatalist is being overlayed over by the second object
    """
    if not isinstance(slicealt,(tuple,list)):
        slicealt=[slicealt]

    xyvecs = [np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = [(None,None),(None,None)]
    title='Neo data and Ne linear interpolation'

    fig3, (ax1, ax2) = subplots(1,2,figsize=(10,5))

    ax1 = GP.alt_slice_overlay((opt, isr), slicealt, xyvecs, vbounds, title, axis=ax1,
                              tind=tind)

    ax2 = GP.alt_contour_overlay((opt, isr), slicealt, xyvecs, vbounds, title, axis=ax2,
                                 tind=tind)

    ax1.set_ylabel('y')
    ax1.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='March 2011 example at PFISR with Neo sCMOS camera')
    p.add_argument('--radaronly',help='load only radar data',action='store_true')
    p.add_argument('-a','--slicealt',help='altitude to take slice',type=float,default=140.)
    p.add_argument('-t','--tind',help='time index(es) to pick',type=int,nargs='+',default=[0])
    p = p.parse_args()

    if p.radaronly:
        optName = None
        azelfn = None
    else:
        optName = '~/data/2011-03-02/110302_0819.h5'
        azelfn = '~/data/2011-03/calMishap2011Mar.h5'

    isr,opt = load_pfisr_hst('~/data/2011-03-02/pfa110302.002.hdf5',optName,azelfn,p.slicealt)

    plotisropt(isr,opt)
    show()

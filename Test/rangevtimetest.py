#!/usr/bin/env python
"""
INCOMPLETE...plots blank graph underneath ax2 handle

@author: John Swoboda
"""
import os
import matplotlib.pyplot as plt
from GeoData.GeoData import GeoData
from GeoData import utilityfuncs
import numpy as np
from GeoData.plotting import rangevstime,insertinfo
import pdb

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def revpower(x1,x2):
    return np.power(x2,x1)


def makeplot():
    figdir = 'Figdump'
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    risrName = 'ran120219.004.hdf5'

    risr_class = GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
    risr_class.changedata('nel','ne',revpower,[10.0])
    (figmplf, [ax1,ax2]) = plt.subplots(2, 1,figsize=(16, 12), facecolor='w')

    vbnd = [5e10,5e11]
    beamnum=1
    beampair = risr_class.dataloc[0,1:]

    rangevstime(risr_class,beamnum,vbnd,'ne',fig=figmplf,ax=ax1)
    rangevstime(risr_class,beampair,vbnd,'ne',fig=figmplf,ax=ax2)

    figname = os.path.join(figdir,'rangevtime.png')
    plt.savefig(figname,format='png',dpi = 600)


if __name__ == "__main__":
    makeplot()

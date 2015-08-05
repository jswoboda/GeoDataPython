#!/usr/bin/env python
"""
quad plot
REQUIRES mayavi for upper left plot (python 2.7 only)

@author: John Swoboda
"""
from __future__ import absolute_import
import os
import matplotlib.pyplot as plt
#
from GeoData.plotting import rangevstime
#
from load_risromti import load_risromti

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def makeplot():
    figdir = 'Figdump'
    try: os.makedirs(figdir)
    except: pass

    risrName = 'ran120219.004.hdf5'
    risr = load_risromti(risrName)[0]
#%%
    (figmplf, [ax1,ax2]) = plt.subplots(2, 1,figsize=(16, 12), facecolor='w')

    vbnd = [5e10,5e11]
    beamnum=1
    beampair = risr.dataloc[0,1:]

    rangevstime(risr,beamnum,vbnd,'ne',fig=figmplf,ax=ax1)
    rangevstime(risr,beampair,vbnd,'ne',fig=figmplf,ax=ax2)

    figname = os.path.join(figdir,'rangevtime.png')
    print('writing ' + figname)
    plt.savefig(figname,format='png',dpi = 600)


if __name__ == "__main__":
    makeplot()

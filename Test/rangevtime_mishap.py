#!/usr/bin/env python
"""
quad plot
REQUIRES mayavi for upper left plot (python 2.7 only)

@author: John Swoboda
"""
from __future__ import division,absolute_import
from matplotlib.pyplot import subplots, show
#
from GeoData.plotting import rangevstime
#
from load_isropt import load_pfisr_neo

def makeplot(isrName):
    risr = load_pfisr_neo(isrName)[0]
#%%
    (figmplf, [ax1,ax2]) = subplots(2, 1,figsize=(16, 10))

    vbnd = [1e10,5e11]
    beamnum=1
    beampair = risr.dataloc[0,1:]

    rangevstime(risr,beamnum,vbnd,'ne',fig=figmplf,ax=ax1)
    rangevstime(risr,beampair,vbnd,'ne',fig=figmplf,ax=ax2)

if __name__ == "__main__":
    makeplot('~/data/2011-03-01/pfa110301.003.hdf5')
    show()

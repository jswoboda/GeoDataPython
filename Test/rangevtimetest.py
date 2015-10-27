#!/usr/bin/env python
"""
@author: John Swoboda
"""
from __future__ import division,absolute_import
from matplotlib.pyplot import figure, show
#
from GeoData.plotting import rangevstime
#
from load_isropt import load_risromti

def makeplot(risrName):
    risr = load_risromti(risrName)[0]
#%%
    fg = figure(figsize=(16, 10))
    ax = fg.gca()

    vbnd = [5e10,5e11]
    beampair = risr.dataloc[0,1:]

    rangevstime(risr,beampair,vbnd,'ne',ax=ax,fig=fg)

if __name__ == "__main__":
    makeplot('~/data/test/ran120219.004.hdf5')
    show()

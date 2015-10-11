#!/usr/bin/env python
"""
quad plot
REQUIRES mayavi for upper left plot (python 2.7 only)

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
    ax = figure(figsize=(16, 10)).gca()

    vbnd = [5e10,5e11]
    beampair = risr.dataloc[0,1:]

    rangevstime(risr,beampair,vbnd,'ne',ax=ax)

if __name__ == "__main__":
    makeplot('ran120219.004.hdf5')
    show()

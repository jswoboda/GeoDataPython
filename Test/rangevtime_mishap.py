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
    isr = load_pfisr_neo(isrName)[0]
#%%
    vbnd = [1e10,5e11]
    beamazel = [-159.5,78.]

    rangevstime(isr,beamazel,vbnd,'ne')

if __name__ == "__main__":
    makeplot('~/data/2011-03-01/pfa110301.003.hdf5')
    show()

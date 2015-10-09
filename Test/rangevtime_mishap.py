#!/usr/bin/env python
"""
quad plot
REQUIRES mayavi for upper left plot (python 2.7 only)

@author: John Swoboda
"""
from __future__ import division,absolute_import
from matplotlib.pyplot import subplots, show
from dateutil.parser import parse
#
from GeoData.plotting import rangevstime
#
from load_isropt import load_pfisr_neo

def makeplot(isrName,tbounds):
    isr = load_pfisr_neo(isrName)[0]
#%%
    vbnd = [1e9,5e11]
    beamazel = [-159.5,78.]

    rangevstime(isr,beamazel,vbnd,'ti',tbounds=tbounds)

if __name__ == "__main__":
    tbounds=(parse('2011-03-01T10:15Z'),
             parse('2011-03-01T11:15Z'))
    makeplot('~/data/2011-03-01/pfa110301.003.hdf5',tbounds)
    show()

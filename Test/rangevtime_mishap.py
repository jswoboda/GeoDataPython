#!/usr/bin/env python3
"""
load isr data vs time and altitude
"""
from __future__ import division,absolute_import
from matplotlib.pyplot import subplots, show
from dateutil.parser import parse
#
from GeoData.plotting import rangevstime
#
from load_isropt import load_pfisr_neo

vbnd = ((1e9,5e11),(500,2500),(500,2500),(-200,200))
beamazel = [[-154.3, 77.5],
            [-149.69,78.56],
            [-159.5, 78.],
            [-154.3, 79.5],
            [-154.3, 278.5]]
cmap = (None,None,None,'bwr')
#titles=('$N_e$','$T_i$','$T_e$','$V_i$')
titles=(None,)*4

def makeplot(isrName,tbounds,isrparams):

    #load radar data into class
    isr = load_pfisr_neo(isrName,isrparams=isrparams)[0]

#%% plot data
    #setup subplot to pass axes handles in to be filled with individual plots
    fg,axs = subplots(5,4,sharex=True,sharey=True,figsize=(16,10))

    for j,(ae,axc) in enumerate(zip(beamazel,axs)):
        for i,(b,p,c,ax,tt) in enumerate(zip(vbnd,isrparams,cmap,axc,titles)):
            rangevstime(isr,ae,b,p[:2],tbounds=tbounds,title=tt,cmap=c,
                        ax=ax,fig=fg,ic=i==0,ir=j==len(axs)-1,it=j==0)

if __name__ == "__main__":
    tbounds=(parse('2011-03-01T10:15Z'),
             parse('2011-03-01T11:15Z'))

    isrparams = ['nel','ti','te','vo']

    makeplot('~/data/2011-03-01/pfa110301.003.hdf5',tbounds,isrparams)
    show()

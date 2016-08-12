#!/usr/bin/env python3
"""
load isr data vs time and altitude
"""
from __future__ import division,absolute_import
import logging
from matplotlib.pyplot import subplots, show,figure,draw,pause
from dateutil.parser import parse
from pytz import UTC
from datetime import datetime
from scipy.spatial import cKDTree
import numpy as np
import seaborn as sns
#
from GeoData.plotting import rangevstime,plotbeamposGD
#
from load_isropt import load_pfisr_neo

epoch = datetime(1970,1,1,tzinfo=UTC)

vbnd = ((1e9,5e11),(500,2500),(500,2500),(-200,200))
beamazel = [[-154.3, 77.5]]
cmap = (None,None,None,'bwr')
#titles=('$N_e$','$T_i$','$T_e$','$V_i$')
titles=(None,)*4

def makeplot(isrName,optName=None,azelfn=None,tbounds=(None,None),isrparams=[None],showbeam=False):

    treq = [(t-epoch).total_seconds() if t else None for t in tbounds ]

    #load radar data into class
    isr,opt = load_pfisr_neo(isrName,optName,azelfn,isrparams=isrparams,treq=treq)

#%% plot data
    #setup subplot to pass axes handles in to be filled with individual plots
    fg,axs = subplots(len(beamazel),4,sharex=True,sharey=True,figsize=(16,10))
    axs = np.atleast_2d(axs)

    for j,(ae,axc) in enumerate(zip(beamazel,axs)):
        for i,(b,p,c,ax,tt) in enumerate(zip(vbnd,isrparams,cmap,axc,titles)):
            rangevstime(isr,ae,b,p[:2],tbounds=tbounds,title=tt,cmap=c,
                        ax=ax,fig=fg,ic=i==0,ir=j==len(axs)-1,it=j==0)
#%%
    plotbeamposGD(isr,minel=75.,elstep=5.)
#%%
    if opt:
        #setup figure
        fg = figure()
        ax = fg.gca()
        hi=ax.imshow(opt.data['optical'][0,...],vmin=50,vmax=250,
                     interpolation='none',origin='lower')
        fg.colorbar(hi,ax=ax)
        ht = ax.set_title('')
        #plot beams
        # find indices of closest az,el
        if showbeam:
            print('building K-D tree for beam scatter plot, takes several seconds')
            kdtree = cKDTree(opt.dataloc[:,1:]) #az,el
            for b in beamazel:
                i = kdtree.query([b[0]%360,b[1]],k=1, distance_upper_bound=0.1)[1]
                y,x = np.unravel_index(i,opt.data['optical'].shape[1:])
                ax.scatter(y,x,s=80,facecolor='none',edgecolor='b')
        #play video
        for t,im in zip(opt.times[:,0],opt.data['optical']):
            hi.set_data(im)
            ht.set_text(datetime.fromtimestamp(t,tz=UTC))
            draw(); pause(0.1)

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='range vs. time plots of key ISR and optical video during March 2011 events')
    p.add_argument('--showbeams',help='superimpose radar beams on video (takes several seconds)',action='store_true')
    p = p.parse_args()

    flist=('~/data/2007-03-23/isr/pfa070301.002.hdf5',
           '~/data/2007-03-23/isr/pfa070301.004.hdf5')

    for f in flist:

        makeplot(f,
                 tbounds=(parse('2007-03-23T00:10Z'), parse('2007-03-23T09:30Z')),
                 isrparams= ['ne','ti','te','vo'],
                 showbeam=p.showbeams)

    show()

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
#
from GeoData.plotting import rangevstime,plotbeamposGD
#
from load_isropt import load_pfisr_neo

epoch = datetime(1970,1,1,tzinfo=UTC)

vbnd = ((1e9,5e11),(500,2500),(500,2500),(-200,200))
beamazel = [[-154.3, 77.5],
            [-149.69,78.56],
            [-159.5, 78.],
            [-154.3, 79.5],
            [-154.3, 78.5]]
cmap = (None,None,None,'bwr')
#titles=('$N_e$','$T_i$','$T_e$','$V_i$')
titles=(None,)*4

def makeplot(isrName,optName,azelfn,tbounds,isrparams):

    treq = (datetime(2011,3,2,8,20,20,tzinfo=UTC),
              datetime(2011,3,2,8,20,21,tzinfo=UTC))

    treq = [(t-epoch).total_seconds() for t in treq]

    #load radar data into class
    isr,opt = load_pfisr_neo(isrName,optName,azelfn,isrparams=isrparams,treq=treq)

#%% plot data
    #setup subplot to pass axes handles in to be filled with individual plots
    fg,axs = subplots(5,4,sharex=True,sharey=True,figsize=(16,10))

    for j,(ae,axc) in enumerate(zip(beamazel,axs)):
        for i,(b,p,c,ax,tt) in enumerate(zip(vbnd,isrparams,cmap,axc,titles)):
            rangevstime(isr,ae,b,p[:2],tbounds=tbounds,title=tt,cmap=c,
                        ax=ax,fig=fg,ic=i==0,ir=j==len(axs)-1,it=j==0)
#%%
    plotbeamposGD(isr,minel=75.,elstep=5.)
#%%
    try:
        fg = figure()
        ax = fg.gca()
        hi=ax.imshow(opt.data['optical'][0,...],vmin=50,vmax=250,
                     interpolation='none',origin='lower')
        fg.colorbar(hi,ax=ax)
        ht = ax.set_title('')
        for t,im in zip(opt.times[:,0],opt.data['optical']):
            hi.set_data(im)
            ht.set_text(datetime.fromtimestamp(t,tz=UTC))
            draw(); pause(0.1)
    except Exception as e:
        logging.error('problem loading images  {}'.format(e))

if __name__ == "__main__":
#%%
    isrparams = ['nel','ti','te','vo']
#%%
    tbounds=(parse('2011-03-02T07:30Z'),
             parse('2011-03-02T09:00Z'))

    makeplot('~/data/2011-03-02/pfa110302.002.hdf5',
             '~/data/2011-03-02/110302_0819.h5',
             '~/data/2011-03/calMishap2011Mar.h5',tbounds,isrparams)
#%%
    if False:
        tbounds=(parse('2011-03-01T10:13Z'),
                 parse('2011-03-01T11:13Z'))

        makeplot('~/data/2011-03-01/pfa110301.003.hdf5',tbounds,isrparams)

    show()

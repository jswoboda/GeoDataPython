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
#
from GeoData.plotting import rangevstime,plotbeamposGD,plotazelscale
#
from load_isropt import load_pfisr_neo

epoch = datetime(1970,1,1,tzinfo=UTC)

vbnd = ((0.5e9,5e11),
        (10,2500),(10,2500),(-200,200))
#beamazel = [[-154.3, 77.5],
#            [-149.69,78.56],
#            [-159.5, 78.],
#            [-154.3, 79.5],
#            [-154.3, 78.5]]
beamazel = np.asarray([[-154.3,77.5]])
cmap = (None,None,None,'bwr')
#titles=('$N_e$','$T_i$','$T_e$','$V_i$')
titles=(None,)*4
SLICEALT=110. #[km]

def makeplot(isrName,optName,azelfn,tbounds,vbounds,isrparams,showbeam,scatterarea):
    """
    inputs:
    -------
    treq: (start,stop) datetime
    """

    treq = [(t-epoch).total_seconds() for t in tbounds]

    #load radar data into class
    isr,opt = load_pfisr_neo(isrName,optName,azelfn,heightkm=SLICEALT,isrparams=isrparams,treq=treq)

#%% plot data
    #setup subplot to pass axes handles in to be filled with individual plots
    fg,axs = subplots(beamazel.shape[0],4,sharex=True,sharey=True,figsize=(16,10))
    axs = np.atleast_2d(axs) #needed for proper sharex in one row case

    for j,ae in enumerate(beamazel):
        for i,(b,p,c,tt,ax) in enumerate(zip(vbnd,isrparams,cmap,titles,axs.ravel())):
            rangevstime(isr,ae,b,p[:2],tbounds=tbounds,title=tt,cmap=c,
                        ax=ax,fig=fg,ic=i==0,ir=j==axs.shape[0]-1,it=j==0)
#%% show ISR beams all alone in az/el plot
    plotbeamposGD(isr) #,minel=75.,elstep=5.
#%% show az/el contours on image
    plotazelscale(opt)
#%% plots optical
    plotoptical(opt,vbounds,showbeam,scatterarea)

def plotoptical(opt,vbounds=(None,None),showbeam=True,scatterarea=80):
    if opt is None:
        return
#%% setup figure
    fg = figure()
    ax = fg.gca()
    hi=ax.imshow(opt.data['optical'][0,...],vmin=vbounds[0],vmax=vbounds[1],
                 interpolation='none',origin='bottom',cmap='gray')
    fg.colorbar(hi,ax=ax)
    ht = ax.set_title('')
    ax.set_axis_off() #no ticks
#%% plot beams over top of video
    if showbeam:  # find indices of closest az,el
        print('building K-D tree for beam scatter plot, takes several seconds')
        kdtree = cKDTree(opt.dataloc[:,1:]) #az,el
        for b in beamazel:
            i = kdtree.query([b[0]%360,b[1]],k=1, distance_upper_bound=0.1)[1]
            y,x = np.unravel_index(i,opt.data['optical'].shape[1:])
            # http://matplotlib.org/examples/color/named_colors.html
            ax.scatter(y,x,s=scatterarea,
                       facecolor='cyan',edgecolor='cyan',
                       alpha=0.4)           #play video
#%% play video
    for t,im in zip(opt.times[:,0],opt.data['optical']):
        hi.set_data(im)
        ht.set_text(datetime.fromtimestamp(t,tz=UTC))
        draw(); pause(0.1)

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='range vs. time plots of key ISR and optical video during March 2011 events')
    p.add_argument('-b','--showbeams',help='superimpose radar beams on video (takes several seconds)',action='store_true')
    p.add_argument('-d','--date',help='date of study event (to auto load files)',required=True)
    p.add_argument('--isr',help='ISR parameters to select',nargs='+',default=['nel','ti','te','vo'])
    p.add_argument('--vlim',help='limits for camera image brightness (contrast adjust)',nargs=2)
    p = p.parse_args()
#%% date / event select
    scatterarea=100 #in case not more accurately specfied vs. fov

    if p.date == '2011-03-02':
        vlim = p.vlim if p.vlim else (50,250)

        tbounds=(datetime(2011,3,2,7,30,tzinfo=UTC),
                 datetime(2011,3,2,9,0,tzinfo=UTC))

        flist=('~/data/2011-03-02/ISR/pfa110302.002.hdf5',
               '~/data/2011-03-02/110302_0819.h5',
               '~/data/2011-03/calMishap2011Mar.h5')

    elif p.date == '2013-04-11':
        vlim = p.vlim if p.vlim else (10,500)

        tbounds=(datetime(2013,4,11,9,tzinfo=UTC),
                 datetime(2013,4,11,12,tzinfo=UTC))

        flist = ('~/data/2013-04-11/ISR/pfa130411.002.hdf5',None,None)

    elif p.date == '2013-04-14_cam0':
        vlim = p.vlim if p.vlim else (250,40000) #it's bright 8:54:25-30 !
        scatterarea = 880

        tbounds=(datetime(2013,4,14,8,54,25,tzinfo=UTC),
                 datetime(2013,4,14,8,54,30,tzinfo=UTC))

        flist = ('~/data/2013-04-14/ISR/pfa130413.004.hdf5',
                 '~/data/2013-04-14/HST/2013-04-14T8-54_hst0.h5',
                 '~/data/2013-04/hst0cal.h5')

    elif p.date == '2013-04-14_cam1':
        vlim = p.vlim if p.vlim else (1050,3500) #it's bright 8:54:25-30 !
        scatterarea = 880

        tbounds=(datetime(2013,4,14,8,54,25,tzinfo=UTC),
                 datetime(2013,4,14,8,54,30,tzinfo=UTC))

        flist = ('~/data/2013-04-14/ISR/pfa130413.004.hdf5',
                 '~/data/2013-04-14/HST/2013-04-14T8-54_hst1.h5',
                 '~/data/2013-04/hst1cal.h5')

    elif p.date == '2013-04-14_dasc':
        vlim = p.vlim if p.vlim else (10,500)

        tbounds=(datetime(2013,4,14,8,0,0,tzinfo=UTC),
                 datetime(2013,4,14,9,0,0,tzinfo=UTC))

        flist = ('~/data/2013-04-14/ISR/pfa130413.004.hdf5',
                 '~/data/2013-04-14/HST/2013-04-14T8-54_hst0.h5',
                 '~/data/2013-04/hst0cal.h5')

    elif p.date=='2013-03-01':
        vlim = p.vlim if p.vlim else (10,500)

        tbounds=(parse('2011-03-01T10:13Z'),
                 parse('2011-03-01T11:13Z'))

        flist = ('~/data/2011-03-01/ISR/pfa110301.003.hdf5',None,None)


    makeplot(flist[0],flist[1],flist[2],tbounds,vlim,p.isr,p.showbeams,scatterarea)
#%%

    show()

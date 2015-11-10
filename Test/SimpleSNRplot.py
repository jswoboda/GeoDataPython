#!/usr/bin/env python3
"""
reading PFISR data down to IQ samples
"""
from __future__ import division
from pathlib2 import Path
from datetime import datetime,timedelta
from pytz import UTC
from dateutil.parser import parse
from numpy import array,log10,absolute, meshgrid,empty,nonzero
from numpy.ma import masked_invalid
import h5py
from pandas import DataFrame
from matplotlib.pyplot import figure,show
from matplotlib.dates import MinuteLocator,SecondLocator
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#sns.color_palette(sns.color_palette("cubehelix"))
#sns.set(context='poster', style='ticks')
#sns.set(rc={'image.cmap': 'cubehelix_r'}) #for contour


def ut2dt(ut):
    if ut.ndim==1:
        T=ut
    elif ut.ndim==2:
        T=ut[:,0]
    return array([datetime.fromtimestamp(t,tz=UTC) for t in T])

def sampletime(T,Np):
    dtime = empty(Np*T.shape[0])
    i=0
    for t in T: #each row
        dt=(t[1]-t[0]) / Np
        for j in range(Np):
            dtime[i]=t[0]+j*dt
            i+=1
    return dtime

def findstride(beammat,bid):
    #FIXME is using just first row OK? other rows were identical for me.
#    Nt = beammat.shape[0]
#    index = empty((Nt,Np),dtype=int)
#    for i,b in enumerate(beammat):
#        index[i,:] = nonzero(b==bid)[0] #NOTE: candidate for np.s_ ?
    return nonzero(beammat[0,:]==bid)[0]

def samplepower(sampiq,bstride,Np,Nr,Nt):
    power = empty((Nr,Np*Nt))
    i=0
    for it in range(Nt):
        for ip in range(Np):
            power[:,i] = (sampiq[it,bstride[ip],:,0]**2 +
                          sampiq[it,bstride[ip],:,1]**2)

            i+=1

    return power


def snrvtime_samples(fn,bid):
    assert isinstance(fn,Path)
    fn = fn.expanduser()

    with h5py.File(str(fn),'r',libver='latest') as f:
        Nt = f['/Time/UnixTime'].shape[0]
        Np = f['/Raw11/Raw/PulsesIntegrated'][0,0] #FIXME is this correct in general?
        ut = sampletime(f['/Time/UnixTime'],Np)
        srng  = f['/Raw11/Raw/Power/Range'].value.squeeze()/1e3
        bstride = findstride(f['/Raw11/Raw/RadacHeader/BeamCode'],bid)
        power = samplepower(f['/Raw11/Raw/Samples/Data'],bstride,Np,srng.size,Nt) #I + jQ   # Ntimes x striped x alt x real/comp

    t = ut2dt(ut)
    return DataFrame(index=srng, columns=t, data=power)



def snrvtime_raw12sec(fn,bid):
    assert isinstance(fn,Path)
    fn = fn.expanduser()

    with h5py.File(str(fn),'r',libver='latest') as f:
        t = ut2dt(f['/Time/UnixTime'])
        bind  = f['/Raw11/Raw/Beamcodes'][0,:] == bid
        power = f['/Raw11/Raw/Power/Data'][:,bind,:].squeeze().T
        srng  = f['/Raw11/Raw/Power/Range'].value.squeeze()/1e3
#%% return requested beam data only
    return DataFrame(index=srng,columns=t,data=power)

def snrvtime_fit(fn,bid):
    assert isinstance(fn,Path)
    fn = fn.expanduser()

    with h5py.File(str(fn),'r',libver='latest') as f:
        t = ut2dt(f['/Time/UnixTime'])
        bind = f['/BeamCodes'][:,0] == bid
        snr = f['/NeFromPower/SNR'][:,bind,:].squeeze().T
        z = f['/NeFromPower/Altitude'][bind,:].squeeze()/1e3
#%% return requested beam data only
    return DataFrame(index=z,columns=t,data=snr)

def plotsnr(snr,fn,tlim=None,vlim=(None,None),zlim=(90,None),ctxt=''):
    assert isinstance(snr,DataFrame)

    fg = figure(figsize=(10,12))
    ax =fg.gca()
    h=ax.pcolormesh(snr.columns.values,snr.index.values,
                     10*log10(masked_invalid(snr.values)),
                     vmin=vlim[0], vmax=vlim[1],cmap='cubehelix_r')
    ax.autoscale(True,tight=True)

    ax.set_xlim(tlim)
    ax.set_ylim(zlim)

    ax.set_ylabel('altitude [km]')
    ax.set_xlabel('Time [UTC]')
#%% date ticks
    fg.autofmt_xdate()
    if tlim:
        tlim[0],tlim[1] = parse(tlim[0]), parse(tlim[1])
        tdiff = tlim[1]-tlim[0]
    else:
        tdiff = snr.columns[-1] - snr.columns[0]

    if tdiff>timedelta(minutes=20):
        ticker = MinuteLocator(interval=5)
    elif (timedelta(minutes=1)<tdiff) & (tdiff<=timedelta(minutes=20)):
        ticker = MinuteLocator(interval=1)
    else:
        ticker = SecondLocator(interval=5)

    ax.xaxis.set_major_locator(ticker)
    ax.tick_params(axis='both', which='both', direction='out')

    c=fg.colorbar(h,ax=ax,fraction=0.075,shrink=0.5)
    c.set_label(ctxt)

    ts = snr.columns[1] - snr.columns[0]
    ax.set_title('{}  {}  $T_{{sample}}$={:.3f} sec.'.format(fn.name, snr.columns[0].strftime('%Y-%m-%d'),ts.microseconds/1e6))


    #last command
    fg.tight_layout()

def plotsnr1d(snr,fn,t0,zlim=(90,None)):
    assert isinstance(snr,DataFrame)
    tind=absolute(snr.columns-t0).argmin()
    tind = range(tind-1,tind+2)
    t1 = snr.columns[tind]

    S = 10*log10(snr.loc[snr.index>=zlim[0],t1])
    z = S.index

    ax = figure().gca()
    ax.plot(S.iloc[:,0],z,color='r')
    ax.plot(S.iloc[:,1],z,color='k')
    ax.plot(S.iloc[:,2],z,color='b')
#    ax.set_ylim(zlim)
    ax.autoscale(True,'y',tight=True)
    ax.set_xlim(-5)

    ax.set_title(fn.name)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('altitude [km]')

def plotsnrmesh(snr,fn,t0,vlim,zlim=(90,None)):
    assert isinstance(snr,DataFrame)
    tind=absolute(snr.columns-t0).argmin()
    tind=range(tind-5,tind+6)
    t1 = snr.columns[tind]

    S = 10*log10(snr.loc[snr.index>=zlim[0],t1])
    z = S.index

    x,y = meshgrid(S.columns.values.astype(float),z)

    ax3 = figure().gca(projection='3d')

#    ax3.plot_wireframe(x,y,S.values)
#    ax3.scatter(x,y,S.values)
    ax3.plot_surface(x,y,S.values,cmap='jet')
    ax3.set_zlim(vlim)
    ax3.set_zlabel('SNR [dB]')
    ax3.set_ylabel('altitude [km]')
    ax3.set_xlabel('time')
    ax3.autoscale(True,'y',tight=True)

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='demo of loading raw ISR data')
    p.add_argument('fn',help='HDF5 file to read')
    p.add_argument('--t0',help='time to extract 1-D vertical plot')
    p.add_argument('--samples',help='use raw samples (lowest level data commnoly available)',action='store_true')
    p.add_argument('--beamid',help='beam id 64157 zenith beam',type=int,default=64157)
    p.add_argument('--vlim',help='min,max for SNR plot [dB]',type=float,nargs=2,default=(35,None))
    p.add_argument('--zlim',help='min,max for altitude [km]',type=float,nargs=2,default=(90,None))
    p.add_argument('--tlim',help='min,max time range yyyy-mm-ddTHH:MM:SSz',nargs=2)
    p = p.parse_args()

#    fn = (Path('~/data/20130413.001_ac_30sec.h5'),
#          Path('~/data/20130413.001_lp_30sec.h5'))
#    t0 = datetime(2013,4,14,8,54,30)
#    for b in ((datetime(2013,4,14,8),   datetime(2013,4,14,10)),
#              (datetime(2013,4,14,8,50),datetime(2013,4,14,9,0))):
#%%
    fn = Path(p.fn).expanduser()
#%% raw (lowest common level)
    if fn.name.endswith('.dt3.h5') and p.samples:
        snrsamp = snrvtime_samples(fn,p.beamid)
        plotsnr(snrsamp,fn,tlim=p.tlim,vlim=p.vlim,ctxt='Power [dB]')
#%% 12 second (numerous integrated pulses)
    elif fn.name.endswith('.dt3.h5'):
        #vlim=(47,None)
        snr12sec = snrvtime_raw12sec(fn,p.beamid)
        plotsnr(snr12sec,fn,vlim=p.vlim,ctxt='SNR [dB]')
#%% 30 second integegration plots
    else:
        #vlim=(-20,None)
        snr = snrvtime_fit(fn,p.beamid)

        if p.t0:
            plotsnr1d(snr,fn,p.t0,p.zlim)
        plotsnr(snr,fn,p.tlim,p.vlim)
        plotsnrmesh(snr,fn,p.t0,p.vlim,p.zlim)

    show()

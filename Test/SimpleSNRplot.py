#!/usr/bin/env python3
from pathlib2 import Path
from datetime import datetime
from numpy import array,log10,absolute, meshgrid
from numpy.ma import masked_invalid
import h5py
from pandas import DataFrame
from matplotlib.pyplot import figure,show
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.color_palette(sns.color_palette("cubehelix"))
sns.set(context='paper', style='whitegrid')
sns.set(rc={'image.cmap': 'cubehelix_r'}) #for contour


def snrvtime(fn,bid):
    fn = Path(fn).expanduser()
    with h5py.File(str(fn),'r',libver='latest') as f:
        t = f[tp].value
        snr = f[dp].value
        bind = f['/BeamCodes'][:,0] == bid
        z = f['/NeFromPower/Altitude'].value

    dt = array([datetime.utcfromtimestamp(T) for T in t[:,0]])
#%% return requested beam data only
    zensnr = snr[:,bind,:].squeeze()
    zenz = z[bind,:].squeeze()/1e3
    return DataFrame(index=zenz,columns=dt,data=zensnr.T)
    #%%

def plotsnr(snr,fn,tlim=None,vlim=(None,None),zlim=(90,None)):
    assert isinstance(snr,DataFrame)
    fn = Path(fn)


    fg = figure()
    ax =fg.gca()
    h=ax.pcolormesh(snr.columns.values,snr.index.values,
                     10*log10(masked_invalid(snr.values)),
                     vmin=vlim[0], vmax=vlim[1])
    ax.autoscale(True,tight=True)

    ax.set_xlim(tlim)
    ax.set_ylim(zlim)

    ax.set_ylabel('altitude [km]')
    ax.set_title(fn.name)
    fg.autofmt_xdate()
    c=fg.colorbar(h,ax=ax)
    c.set_label('SNR [dB]')

def plotsnr1d(snr,fn,t0,zlim):
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

def plotsnrmesh(snr,fn,vlim):

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
    fn = (Path('~/data/20130413.001_ac_30sec.h5'),
          Path('~/data/20130413.001_lp_30sec.h5'))
    dp = '/NeFromPower/SNR'
    tp = '/Time/UnixTime'
    beamid = 64157 #zenith
    vlim = (-20,None)
    t0 = datetime(2013,4,14,8,54,30)
    zlim=(90,None)
#%%
    for f in fn:
        snr = snrvtime(f,beamid)
#        plotsnr1d(snr,f,t0,zlim)
        for b in ((datetime(2013,4,14,8),   datetime(2013,4,14,10)),
                  (datetime(2013,4,14,8,50),datetime(2013,4,14,9,0))):
#            plotsnr(snr,f,b,vlim)
            plotsnrmesh(snr,f,vlim)

    show()

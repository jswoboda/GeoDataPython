#!/usr/bin/env python3
from pathlib2 import Path
from datetime import datetime
from numpy import array,log10
from numpy.ma import masked_invalid
import h5py
from pandas import DataFrame
from matplotlib.pyplot import figure,show
from matplotlib.colors import LogNorm

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

def plotsnr(snr,fn,tlim=None,vlim=(None,None)):
    assert isinstance(snr,DataFrame)
    fn = Path(fn)


    fg = figure()
    ax =fg.gca()
    h=ax.pcolormesh(snr.columns.values,snr.index.values,
                     10*log10(masked_invalid(snr.values)),
                     vmin=vlim[0], vmax=vlim[1])
    ax.autoscale(True,tight=True)

    ax.set_xlim(tlim)
    ax.set_ylim(90,None)

    ax.set_ylabel('altitude [km]')
    ax.set_title(fn.name)
    fg.autofmt_xdate()
    c=fg.colorbar(h,ax=ax)
    c.set_label('SNR [dB]')

if __name__ == '__main__':
    fn = ('~/data/20130413.001_ac_30sec.h5',
          '~/data/20130413.001_lp_30sec.h5')
    dp = '/NeFromPower/SNR'
    tp = '/Time/UnixTime'
    beamid = 64157 #zenith
    vlim = (-10,None)
#%%
    for f in fn:
        snr = snrvtime(f,beamid)
        for b in ((datetime(2013,4,14,8),   datetime(2013,4,14,10)),
                  (datetime(2013,4,14,8,50),datetime(2013,4,14,9,0))):
            plotsnr(snr,f,b,vlim)

    show()

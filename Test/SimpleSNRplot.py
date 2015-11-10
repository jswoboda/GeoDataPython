#!/usr/bin/env python3
from __future__ import division, absolute_import
from os.path import expanduser
from datetime import datetime
from numpy import array,log10
import h5py
from matplotlib.pyplot import figure,show
from matplotlib.colors import LogNorm

fn = '~/data/20130413.001_ac_30sec.h5'
dp = '/NeFromPower/SNR'
tp = '/Time/UnixTime'
bid = 64157 #zenith

with h5py.File(expanduser(fn),'r',libver='latest') as f:
    t = f[tp].value
    snr = f[dp].value
    bind = f['/BeamCodes'][:,0] == bid
    z = f['/NeFromPower/Altitude'].value

zensnr = snr[:,bind,:].squeeze()
zenz = z[bind,:].squeeze()/1e3
#%%
dt = array([datetime.utcfromtimestamp(T) for T in t[:,0]])

fg = figure()
ax =fg.gca()
h=ax.pcolormesh(dt,zenz,10*log10(zensnr.T),)
             # norm=LogNorm())
ax.autoscale(True,tight=True)
ax.set_xlim(datetime(2013,4,14,8),datetime(2013,4,14,10))
ax.set_ylim(90,None)
ax.set_ylabel('altitude [km]')
fg.autofmt_xdate()
c=fg.colorbar(h,ax=ax)
c.set_label('SNR [dB]')


show()
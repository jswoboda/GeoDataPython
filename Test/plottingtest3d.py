#!/usr/bin/env python
"""
@author: John Swoboda
"""
from __future__ import division,print_function
import logging
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np
#
from GeoData.plotting import slice2DGD, plotbeamposGD, insertinfo, contourGD
from GeoData.plottingmayavi import plot3Dslice
#
from load_isropt import load_risromti
#
try:
    from mayavi import mlab
except ImportError:
    pass

omtislices =  [[],[],[140]]
risrslices = [[100],[300],[]]
vbounds = [[200,800],[5e10,5e11]]

def make_data(risrName,omtiName):

    risr,omti = load_risromti(risrName,omtiName)
#%% creating GeoData objects of the 2 files, given a specific parameter
    reglistlist = omti.timeregister(risr)
    keepomti = [i  for i in range(len(reglistlist)) if len(reglistlist[i])>0]
    reglist = list(set(np.concatenate(reglistlist)))
#%% converting logarthmic electron density (nel) array into electron density (ne) array
    risr_classred =risr.timeslice(reglist,'Array')
    omti = omti.timeslice(keepomti,'Array')
    reglistfinal = omti.timeregister(risr_classred)
    xvec,yvec,zvec = [np.linspace(-100.0,500.0,25),np.linspace(0.0,600.0,25),np.linspace(100.0,500.0,25)]
    x,y,z = np.meshgrid(xvec,yvec,zvec)
    x2d,y2d = np.meshgrid(xvec,yvec)
    new_coords =np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    new_coords2 = np.column_stack((x2d.ravel(),y2d.ravel(),140.0*np.ones(y2d.size)))
    #%% interpolate risr data
    risr_classred.interpolate(new_coords, newcoordname='Cartesian', method='linear', fill_value=np.nan)
    #%% interpolate omti data
    omti.interpolate(new_coords2, newcoordname='Cartesian', twodinterp = True,method='linear', fill_value=np.nan)
    return (risr_classred,risr,omti,reglistfinal)

def plotting(risr_classred,risr_class,omti_class,reglistfinal,odir=None):

    figcount = 0
    for iomti,ilist in enumerate(reglistfinal):
        for irisr in ilist:
            omtitime = iomti
            risrtime =  irisr

            try:
                mfig = mlab.figure(fgcolor=(1, 1, 1), bgcolor=(1, 1, 1))
                surflist1 = plot3Dslice(omti_class, omtislices, vbounds[0],
                            time = omtitime,cmap='gray',gkey = 'optical',fig=mfig)
                arr = plot3Dslice(risr_classred, risrslices, vbounds[1],
                            time = risrtime,cmap='jet',gkey = 'ne',fig=mfig,units = 'm^{-3}',colorbar = True,view=[50,60],outimage=True)
                titlestr1 = '$N_e$ and OMTI at $thm'
                newtitle = insertinfo(titlestr1,'',risr_classred.times[risrtime,0],risr_classred.times[risrtime,1])
            except Exception as e:
                logging.error('trouble with 3D plot  {}'.format(e))

            (figmplf, [[ax1,ax2],[ax3,ax4]]) = plt.subplots(2, 2,figsize=(16, 12), facecolor='w')

            try:
                ax1.imshow(arr)
                ax1.set_title(newtitle)
                ax1.axis('off')
            except:
                pass

            (slice2,cbar2) = slice2DGD(risr_classred,'z',400,vbounds[1],title='$N_e$ at $thm',
                        time = risrtime,cmap='jet',gkey = 'ne',fig=figmplf,ax=ax2)
            cbar2.set_label('$N_e$ in $m^{-3}$')

            (slice3,cbar3) = slice2DGD(omti_class,'z',omtislices[-1][0],vbounds[0],title='OMTI at $thm',
                        time = omtitime,cmap='Greys',gkey = 'optical',fig=figmplf,ax=ax3,cbar=False)
            plt.hold(True)

            (slice4,cbar4) = contourGD(risr_classred,'z',400,vbounds[1],title='$N_e$ at $thm',
                        time = risrtime,cmap='jet',gkey = 'ne',fig=figmplf,ax=ax3)
            cbar4.set_label('$N_e$ in $m^{-3}$')
            bmpos = plotbeamposGD(risr_class)

            if odir is not None:
                figname = os.path.join(odir,'figure{0:0>2}.png'.format(figcount))
                figmplf.savefig(figname,format='png',dpi = 400)
                plt.close(figmplf)
                figcount=figcount+1

if __name__ == "__main__":
    (risr_classred,risr_class,omti_class,reglistfinal) = make_data('data/ran120219.004.hdf5','data/OMTIdata.h5')
    plotting(risr_classred,risr_class,omti_class,reglistfinal)

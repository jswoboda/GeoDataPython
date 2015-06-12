#!/usr/bin/env python
"""
INCOMPLETE...plots blank graph underneath ax2 handle

@author: John Swoboda
"""
import os
import matplotlib.pyplot as plt
from GeoData.GeoData import GeoData
from GeoData import utilityfuncs
import numpy as np
from GeoData.plotting import slice2DGD,plot3Dslice,plotbeamposGD,insertinfo
from mayavi import mlab
import pdb

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def revpower(x1,x2):
    return np.power(x2,x1)


def make_data():
    risrName = 'ran120219.004.hdf5'
    omtiName = 'OMTIdata.h5'

    #creating GeoData objects of the 2 files, given a specific parameter
    omti_class = GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
    risr_class = GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))

    reglistlist = omti_class.timeregister(risr_class)
    keepomti = [i  for i in range(len(reglistlist)) if len(reglistlist[i])>0]
    reglist = list(set(np.concatenate(reglistlist)))
    #converting logarthmic electron density (nel) array into electron density (ne) array
    risr_class.changedata('nel','ne',revpower,[10.0])
    risr_classred =risr_class.timeslice(reglist,'Array')
    omti_class = omti_class.timeslice(keepomti,'Array')
    reglistfinal = omti_class.timeregister(risr_classred)
    xvec,yvec,zvec = [np.linspace(-100.0,500.0,25),np.linspace(0.0,600.0,25),np.linspace(100.0,500.0,25)]
    x,y,z = np.meshgrid(xvec,yvec,zvec)
    x2d,y2d = np.meshgrid(xvec,yvec)
    new_coords =np.column_stack((x.flatten(),y.flatten(),z.flatten()))
    new_coords2 = np.column_stack((x2d.flatten(),y2d.flatten(),140.0*np.ones(y2d.size)))
    #interpolate risr data
    risr_classred.interpolate(new_coords, newcoordname='Cartesian', method='linear', fill_value=np.nan)
    # interpolate omti data
    omti_class.interpolate(new_coords2, newcoordname='Cartesian', twodinterp = True,method='linear', fill_value=np.nan)
    return (risr_classred,risr_class,omti_class,reglistfinal)
def plotting(risr_classred,risr_class,omti_class,reglistfinal):
    #path names to h5 files
    figdir = 'Figdump'
    figcount =0
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    for iomti,ilist in enumerate(reglistfinal):
        for irisr in ilist:
            mfig = mlab.figure(fgcolor=(1, 1, 1), bgcolor=(1, 1, 1))
            omtitime = iomti
            risrtime =  irisr
            omtislices =  [[],[],[140]]
            risrslices = [[100],[300],[]]
            vbounds = [[200,800],[5e10,5e11]]
            surflist1 = plot3Dslice(omti_class, omtislices, vbounds[0],
                        time = omtitime,cmap='gray',gkey = 'optical',fig=mfig)
            arr = plot3Dslice(risr_classred, risrslices, vbounds[1],
                        time = risrtime,cmap='jet',gkey = 'ne',fig=mfig,units = 'm^{-3}',colorbar = True,outimage=True)
            titlestr1 = '$N_e$ and OMTI at $thm'
            newtitle = insertinfo(titlestr1,'',risr_classred.times[risrtime,0],risr_classred.times[risrtime,1])

            (figmplf, [[ax1,ax2],[ax3,ax4]]) = plt.subplots(2, 2,figsize=(16, 12), facecolor='w')

            ax1.imshow(arr)
            ax1.set_title(newtitle)
            ax1.set_title(newtitle)
            ax1.axis('off')

            slice2 = slice2DGD(risr_classred,'z',400,vbounds[1],title='$N_e$ at $thm',
                        time = risrtime,cmap='jet',gkey = 'ne',fig=figmplf,ax=ax2)

            slice3 = slice2DGD(omti_class,'z',omtislices[-1][0],vbounds[0],title='OMTI at $thm',
                        time = omtitime,cmap='Greys',gkey = 'optical',fig=figmplf,ax=ax3)

            bmpos = plotbeamposGD(risr_class,fig=figmplf,ax=ax4)
            figname = os.path.join(figdir,'figure{0:0>2}.png'.format(figcount))
            plt.savefig(figname,format='png',dpi = 600)
            plt.close(figmplf)
            figcount = figcount+1
if __name__ == "__main__":
    (risr_classred,risr_class,omti_class,reglistfinal) = make_data()
    plotting(risr_classred,risr_class,omti_class,reglistfinal)
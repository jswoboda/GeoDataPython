# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: Anna Stuhlmacher

plotting
"""
from __future__ import division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as spinterp
import time
import pdb
from warnings import warn
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import matplotlib as mpl
#from matplotlib import ticker
try:
    from mayavi import mlab
except Exception as e:
    warn('could not import Mayavi. Some 3-D plots will be disabled.  {}'.format(e))

try:
    from .CoordTransforms import angles2xy
except:
    from CoordTransforms import angles2xy


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title, axis=None):
    """
    geodatalist - A list of geodata objects that will be overlayed, first object is on the bottom and in gray scale
    altlist - A list of the altitudes that we can overlay.
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for each geodata object. ie, vbounds=[[500,2000], [5e10,5e11]]
    title - A string that holds for the overall image
    Returns an image of an overlayed plot at a specific altitude.
    """
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    x,y = np.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    new_coords = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    key0 = list(geodatalist[0].data.keys()) #list necessary for Python3
    key1 = list(geodatalist[1].data.keys())

    gd2 = geodatalist[1].timeslice([1,2]) #second and third times in array
    gd2.interpolate(new_coords, newcoordname='Cartesian', method='linear', fill_value=np.nan)
    interpData = gd2.data[key1[0]]
    risr = interpData[:,0].reshape(x.shape)

    gd3 = geodatalist[0].timeslice([1,2])
    gd3.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd3.data[key0[0]]
    omti = interpData[:,0].reshape(x.shape)

    if axis is None:
        fg = plt.figure(facecolor='white'); ax=fg.gca()
        bottom = ax.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        cbar1 = fg.colorbar(bottom)
        cbar1.set_label(key0[0])
        ax.hold(True)

        top = ax.imshow(risr, cmap=cm.jet, alpha=0.4, extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        cbar2 = fg.colorbar(top)
        cbar2.set_label(key1[0])
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    else:
        axis.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        axis.hold(True)
        axis.imshow(risr, cmap=cm.jet, alpha=0.4, extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        return axis

def alt_contour_overlay(geodatalist, altlist, xyvecs, vbounds, title, axis=None):
    """
    geodatalist - A list of geodata objects that will be overlayed, first object is on the bottom and in gray scale
    altlist - A list of the altitudes that we can overlay.
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over.
    vbounds = a list of bounds for each geodata object. ie, vbounds=[[500,2000], [5e10,5e11]]
    title - A string that holds for the overall image
    Returns an image of an overlayed plot at a specific altitude.
    """
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    x,y = np.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    new_coords = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    key0 = list(geodatalist[0].data.keys()) #list needed for Python3
    key1 = list(geodatalist[1].data.keys())

    gd2 = geodatalist[1].timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd2.data[key1[0]]
    risr = interpData[:,0].reshape(x.shape)

    gd3 = geodatalist[0].timeslice([1,2])
    gd3.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd3.data[key0[0]]
    omti = interpData[:,0].reshape(x.shape)

    if axis == None:
        fg= plt.figure(facecolor='white'); ax=fg.gca()
        bottom = ax.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        cbar1 = plt.colorbar(bottom, orientation='horizontal')
        cbar1.set_label(key0[0])
        ax.hold(True)

        top = ax.contour(x,y, risr, cmap=cm.jet,extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        #clabel(top,inline=1,fontsize=10, fmt='%1.0e')
        cbar2 = fg.colorbar(top, format='%.0e')
        cbar2.set_label(key1[0])
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        axis.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        axis.hold(True)
        axis.contour(x,y, risr, cmap=cm.jet,extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        return axis



def plot3Dslice(geodata,surfs,vbounds, titlestr='', time = 0,gkey = None,cmap='jet', ax=None,fig=None,method='linear',fill_value=np.nan,view = None,units='',colorbar=False,outimage=False):
    """ This function create 3-D slice image given either a surface or list of coordinates to slice through
    Inputs:
    geodata - A geodata object that will be plotted in 3D
    surfs - This is a three element list. Each element can either be
    altlist - A list of the altitudes that RISR parameter slices will be taken at
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for the geodata objec's parameters. ie, vbounds=[500,2000]
    title - A string that holds for the overall image
    ax - A handle for an axis that this will be plotted on.

    Returns a mayavi image with a surface
    """
    assert geodata.coordnames.lower() =='cartesian'

    datalocs = geodata.dataloc

    xvec = sp.unique(datalocs[:,0])
    yvec = sp.unique(datalocs[:,1])
    zvec = sp.unique(datalocs[:,2])

    assert len(xvec)*len(yvec)*len(zvec)==datalocs.shape[0]

    #determine if the ordering is fortran or c style ordering
    diffcoord = sp.diff(datalocs,axis=0)

    if diffcoord[0,1]!=0.0:
        ord='f'
    elif diffcoord[0,2]!=0.0:
        ord='c'
    elif diffcoord[0,0]!=0.0:
        if len(np.where(diffcoord[:,1])[0])==0:
            ord = 'f'
        elif len(np.where(diffcoord[:,2])[0])==0:
            ord = 'c'

    matshape = (len(yvec),len(xvec),len(zvec))
    # reshape the arrays into a matricies for plotting
    x,y,z = [sp.reshape(datalocs[:,idim],matshape,order=ord) for idim in range(3)]

    if gkey is None:
        gkey = geodata.datanames()[0]
    porig = geodata.data[gkey][:,time]

    mlab.figure(fig)
    #determine if list of slices or surfaces are given

    islists = type(surfs[0])==list
    if type(surfs[0])==np.ndarray:
        onedim = surfs[0].ndim==1
    #get slices for each dimension out
    surflist = []
    if islists or onedim:
        p = np.reshape(porig,matshape,order= ord )
        xslices = surfs[0]
        for isur in xslices:
            indx = sp.argmin(sp.absolute(isur-xvec))
            xtmp = x[:,indx]
            ytmp = y[:,indx]
            ztmp = z[:,indx]
            ptmp = p[:,indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)]=True
            surflist.append( mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap,mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0

        yslices = surfs[1]
        for isur in yslices:
            indx = sp.argmin(sp.absolute(isur-yvec))
            xtmp = x[indx]
            ytmp = y[indx]
            ztmp = z[indx]
            ptmp = p[indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)]=True
            surflist.append( mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap,mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
        zslices = surfs[2]
        for isur in zslices:
            indx = sp.argmin(sp.absolute(isur-zvec))
            xtmp = x[:,:,indx]
            ytmp = y[:,:,indx]
            ztmp = z[:,:,indx]
            ptmp = p[:,:,indx]
            pmask = sp.zeros_like(ptmp).astype(bool)
            pmask[sp.isnan(ptmp)]=True
            surflist.append( mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap,mask=pmask))
            surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    else:
        # For a general surface.
        xtmp,ytmp,ztmp = surfs[:]
        gooddata = ~np.isnan(porig)
        curparam = porig[gooddata]
        curlocs = datalocs[gooddata]
        new_coords = np.column_stack((xtmp.flatten(),ytmp.flatten(),ztmp.flatten()))
        ptmp = spinterp.griddata(curlocs,curparam,new_coords,method,fill_value)
        pmask = sp.zeros_like(ptmp).astype(bool)
        pmask[sp.isnan(ptmp)]=True
        surflist.append( mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap,mask=pmask))
        surflist[-1].module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    mlab.title(titlestr,color=(0,0,0))
    #mlab.outline(color=(0,0,0))
    mlab.axes(color=(0,0,0),x_axis_visibility=True,xlabel = 'x in km',y_axis_visibility=True,
              ylabel = 'y in km',z_axis_visibility=True,zlabel = 'z in km')

    mlab.orientation_axes(xlabel = 'x in km',ylabel = 'y in km',zlabel = 'z in km')

    if colorbar:
        if len(units)>0:
            titlstr = gkey +' in ' +units
        else:
            titlestr = gkey
        mlab.colorbar(surflist[-1],title=titlstr,orientation='vertical')
    if outimage:
        arr = mlab.screenshot(fig,antialiased = True)
        mlab.close(fig)
        return arr
    else:
        return surflist

def plotbeamposGD(geod,fig=None,ax=None,title='Beam Positions'):
        assert geod.coordnames.lower() =='spherical'

        if (ax is None) and (fig is None):
            fig = plt.figure(facecolor='white')
            ax = fig.gca()
        elif ax is None:
            ax = fig.gca()

        make_polax(fig,ax,False)
        plt.hold(True)
        (azvec,elvec) = (geod.dataloc[:,1],geod.dataloc[:,2])

        (xx2,yy2) = angles2xy(azvec,elvec,False)
        plotsout = plt.plot(xx2,yy2,'o',c='b', markersize=10)
        plt.title(title)
        return plotsout
def make_polax(fig,ax,zenith):
    """ This makes the polar axes for the beams"""
    if zenith:
        minel = 0.0
        maxel = 70.0
        elspace = 10.0
        ellines = np.arange(minel,maxel,elspace)
    else:
        minel = 30.0
        maxel = 90.0
        elspace = 10.0
        ellines = np.arange(minel,maxel,elspace)

    azlines = np.arange(0.0,360.0,30.0)

    # plot all of the azlines
    elvec = np.linspace(maxel,minel,100)
    for iaz in azlines:
        azvec = iaz*np.ones_like(elvec)
        (xx,yy) = angles2xy(azvec,elvec,zenith)
        plt.plot(xx,yy,'k--')
        plt.hold(True)
        (xt,yt) = angles2xy(azvec[-1],elvec[-1]-5,zenith)
        plt.text(xt,yt,str(int(iaz)))

    azvec = np.linspace(0.0,360,100)
    # plot the el lines
    for iel in ellines:
        elvec = iel*np.ones_like(azvec)
        (xx,yy) = angles2xy(azvec,elvec,zenith)
        plt.plot(xx,yy,'k--')
        (xt,yt) = angles2xy(315,elvec[-1]-3,zenith)
        plt.text(xt,yt,str(int(iel)))
    plt.axis([-90,90,-90,90])
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)


def slice2DGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,ax=None,title='',units=''):

    #xyzvecs is the area that the data covers.

    assert geod.coordnames.lower() =='cartesian'

    axdict = {'x':0,'y':1,'z':2}
    veckeys = ['x','y','z']
    if type(axstr)==str:
        axis=axstr
    else:
        axis= veckeys[axstr]
    veckeys.remove(axis.lower())
    datacoords = geod.dataloc
    xyzvecs = {l:sp.unique(datacoords[:,axdict[l]]) for l in veckeys}
    veckeys.sort()
    #make matrices
    M1,M2 = sp.meshgrid(xyzvecs[veckeys[0]],xyzvecs[veckeys[1]])
    slicevec = sp.unique(datacoords[:,axdict[axis]])
    min_idx = sp.argmin(sp.absolute(slicevec-slicenum))
    slicenum=slicevec[min_idx]
    rec_coords = {axdict[veckeys[0]]:M1.flatten(),axdict[veckeys[1]]:M2.flatten(),
                  axdict[axis]:slicenum*sp.ones(M2.size)}


    new_coords = sp.zeros((M1.size,3))
    #make coordinates
    for ckey in rec_coords.keys():
        new_coords[:,ckey] = rec_coords[ckey]
    #determine the data name
    if gkey is None:
        gkey = geod.data.keys[0]
    # get the data location
    dataout = geod.datareducelocation(new_coords,'Cartesian',gkey)[:,time]


    title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])
    dataout = sp.reshape(dataout,M1.shape)

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    ploth = ax.pcolor(M1,M2,dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)
    ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(), xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
    cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
    cbar2.set_label(gkey+' in ' +units)
    ax.set_title(title)
    ax.set_xlabel(veckeys[0])
    ax.set_ylabel(veckeys[1])

    return(ploth)


def insertinfo(strin,key='',posix=None,posixend = None):

    listin = type(strin)==list
    if listin:
        stroutall = []
    else:
        strin=[strin]

    for k in range(len(strin)):

        strout = strin[k].replace('$k',key)
        if posix is None:
            strout.replace()
            strout=strout.strip('$tu')
            strout=strout.strip('$tdu')
        else:
            curdt = time.gmtime(posix);
            curdte = time.gmtime(posixend);
            markers = [
                '$thmsehms',#UT hours minutes seconds - hours minutes seconds
                '$thmehm',#UT hours minutes - hours minutes
                '$tmsems',#UT minutes seconds - minutes seconds
                '$thms',#UT hours minutes seconds
                '$thm',#UT hours minutes
                '$tms',#UT minutes seconds
                '$tmdyhms',#UT month/day/year hours minutes seconds
                '$tmdyhm',#UT month/day/year hours minutes
                '$tmdhm'#UT month/day hours minutes
                ]
            datestrcell = [
                time.strftime('%H:%M:%S',curdt)+' - '+time.strftime('%H:%M:%S',curdte)+' UT',
                time.strftime('%H:%M',curdt)+' - '+time.strftime('%H:%M',curdte)+' UT',
                time.strftime('%M:%S',curdt)+' - '+time.strftime('%M:%S',curdte)+' UT',
                time.strftime('%H:%M:%S',curdt)+' UT',
                time.strftime('%H:%M',curdt)+' UT',
                time.strftime('%M:%S',curdt)+' UT',
                time.strftime('%m/%d/%Y %H:%M:%S',curdt)+' UT',
                time.strftime('%m/%d/%Y %H:%M',curdt)+' UT',
                time.strftime('%m/%d %H:%M',curdt)+' UT']
            for imark in range(len(markers)):
                strout=strout.replace(markers[imark],datestrcell[imark]);

        if listin:
            stroutall[k] = strout
        else:
            stroutall = strout
    return stroutall
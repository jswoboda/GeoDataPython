# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: Anna Stuhlmacher

plotting
"""
from __future__ import division, absolute_import
import logging
import numpy as np
import scipy as sp
import scipy.interpolate as spinterp
import time
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import ScalarFormatter
import pdb
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib import ticker
try:
    from mayavi import mlab
except Exception as e:
    pass
#
from .CoordTransforms import angles2xy,sphereical2Cartisian
from .GeoData import GeoData
# NOTE: using usetex can make complicated plots unstable and crash
#try:
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#except Exception as e:
#    logging.info('Latex install not complete, falling back to basic fonts.  sudo apt-get install dvipng')
#
sfmt = ScalarFormatter(useMathText=True)
#%%
def _dointerp(geodatalist,altlist,xyvecs,picktimeind):
    opt=None; isr=None #in case of failure
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    x,y = np.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    new_coords = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    key={}
#%% iterative demo, not used yet
#    inst = []
#    for g in geodatalist:
#        if g is None:
#            continue
#        for k in g.data.keys():
#            try:
#                G = g.timeslice(picktimeind)
#                G.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
#                interpData = G.data[k]
#                inst.append(interpData[:,0].reshape(x.shape))
#            except Exception as e:
#                logging.warning('skipping instrument   {}'.format(e))
#%% optical
    g = geodatalist[0]
    try:
        key['opt'] = list(g.data.keys()) #list necessary for Python3
        G = g.timeslice(picktimeind)
        G.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
        interpData = G.data[key['opt'][0]]
        opt = interpData[:,0].reshape(x.shape)
    except IndexError as e:
        logging.warning('did you pick a time index outside camera observation?  {}'.format(e))
    except Exception as e:
        logging.error('problem in optical interpolation   {}'.format(e))
#%% isr
    g = geodatalist[1]
    try:
        key['isr'] = list(g.data.keys()) #list necessary for Python3
        G = g.timeslice(picktimeind)
        G.interpolate(new_radar_coords, newcoordname='Cartesian', method='nearest', 
                      fill_value=np.nan)
        interpData = G.data[key['isr'][0]]
        isr = interpData[:,0].reshape(x.shape)
    except Exception as e:
        logging.error('problem in ISR interpolation   {}'.format(e))

    return opt,isr,extent,key,x,y
#%%
def alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title, axis=None,picktimeind=[0]):
    """
    geodatalist - A list of geodata objects that will be overlayed, first object is on the bottom and in gray scale
    altlist - A list of the altitudes that we can overlay.
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for each geodata object. ie, vbounds=[[500,2000], [5e10,5e11]]
    title - A string that holds for the overall image
    picktimeind - indices in time to extract and plot (arbitrary choice)
    Returns an image of an overlayed plot at a specific altitude.
    """
    ax = axis #less typing
    opt,isr,extent,key,x,y = _dointerp(geodatalist,altlist,xyvecs,picktimeind)
#%% plots
    if ax is None:
        fg = plt.figure()
        ax=fg.gca()
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        fg = ax.get_figure()
#%%
    try:
        bottom = ax.imshow(opt, cmap='gray', extent=extent, origin='lower', interpolation='none',
                           vmin=vbounds[0][0],vmax=vbounds[0][1])
        c = fg.colorbar(bottom,ax=ax)
        c.set_label(key['opt'][0])
    except Exception as e:
        logging.info('problem plotting Optical slice {}'.format(e))
#%%
    if isr is None or not np.isfinite(isr).any():
        logging.warning('Nothing to plot for ISR, all NaN')

    try:
        top = ax.imshow(isr, alpha=0.4, extent=extent, origin='lower',interpolation='none',
                        vmin=vbounds[1][0],vmax=vbounds[1][1])
        c = fg.colorbar(top,ax=ax)
        c.set_label(key['isr'][0])
    except Exception as e:
        logging.info('problem plotting ISR slice {}'.format(e))

    return ax
#%%
def alt_contour_overlay(geodatalist, altlist, xyvecs, vbounds, title, axis=None,picktimeind=[1,2]):
    """
    geodatalist - A list of geodata objects that will be overlayed, first object is on the bottom and in gray scale
    altlist - A list of the altitudes that we can overlay.
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over.
    vbounds = a list of bounds for each geodata object. ie, vbounds=[[500,2000], [5e10,5e11]]
    title - A string that holds for the overall image
    picktimeind - indices in time to extract and plot (arbitrary choice)
    Returns an image of an overlayed plot at a specific altitude.
    """
    ax = axis #less typing
    opt,isr,extent,key,x,y = _dointerp(geodatalist,altlist,xyvecs,picktimeind)
#%% plots
    if axis is None:
        fg= plt.figure()
        ax=fg.gca()
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        fg = ax.get_figure()
#%%
    try:
        bottom = ax.imshow(opt, cmap='gray', extent=extent, origin='lower', interpolation='none',
                           vmin=vbounds[0][0],vmax=vbounds[0][1])
        cbar1 = plt.colorbar(bottom, orientation='horizontal',ax=ax)
        cbar1.set_label(key['opt'][0])
    except Exception as e:
        logging.info('problem plotting optical   {}'.format(e))

    try:
        top = ax.contour(x,y, isr,extent=extent, origin='lower', interpolation='none',
                         vmin=vbounds[1][0],vmax=vbounds[1][1])
        #clabel(top,inline=1,fontsize=10, fmt='%1.0e')
        cbar2 = fg.colorbar(top, format='%.0e',ax=ax)
        cbar2.set_label(key['isr'][0])
    except Exception as e:
        logging.info('problem plotting isr contour {}'.format(e))

    return ax


#%%
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

    islists = isinstance(surfs[0],list)
    if isinstance(surfs[0],np.ndarray):
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

    if view is not None:
        mlab.view(view[0],view[1])
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

def slice2DGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,
              ax=None,title='',cbar=True,m=None):

    #xyzvecs is the area that the data covers.
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}# shows which row is this coordinate
        veckeys = ['long','lat','alt']# shows which is the x, y and z axes for plotting

    if type(axstr)==str:
        axis=axstr
    else:
        axis= veckeys[axstr]
    veckeys.remove(axis.lower())
    veckeys.append(axis.lower())
    datacoords = geod.dataloc
    xyzvecs = {l:sp.unique(datacoords[:,axdict[l]]) for l in veckeys}

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

    # get the data location, first check if the data can be just reshaped then do a
    # search

    sliceindx = slicenum==datacoords[:,axdict[axis]]

    datacoordred = datacoords[sliceindx]
    rstypes = ['C','F','A']
    nfounds = True
    M1dlfl = datacoordred[:,axdict[veckeys[0]]]
    M2dlfl = datacoordred[:,axdict[veckeys[1]]]
    for ir in rstypes:
        M1dl = sp.reshape(M1dlfl,M1.shape,order =ir)
        M2dl = sp.reshape(M2dlfl,M1.shape,order =ir)
        if sp.logical_and(sp.allclose(M1dl,M1),sp.allclose(M2dl,M2)):
            nfounds=False
            break
    if nfounds:
        dataout = geod.datareducelocation(new_coords,geod.coordnames,gkey)[:,time]
        dataout = sp.reshape(dataout,M1.shape)
    else:
        dataout = sp.reshape(geod.data[gkey][sliceindx,time],M1.shape,order=ir)

    title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])


    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()
    if m is None:
        ploth = ax.pcolor(M1,M2,dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap,
                          linewidth=0,rasterized=True)
        ploth.set_edgecolor('face')
        ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(),
                 xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
        if cbar:
            cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
        else:
            cbar2 = None
        ax.set_title(title)
        ax.set_xlabel(veckeys[0])
        ax.set_ylabel(veckeys[1])
    else:
        N1,N2 = m(M1,M2)
        ploth = m.pcolor(N1,N2,dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap,
                         alpha=.4,linewidth=0,rasterized=True)

        if cbar:
            cbar2 = m.colorbar(ploth, format='%.0e')
        else:
            cbar2 = None


    return(ploth,cbar2)

def contourGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',
              fig=None,ax=None,title='',cbar=True,m=None,levels=None):
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}# shows which row is this coordinate
        veckeys = ['long','lat','alt']# shows which is the x, y and z axes for plotting
    if type(axstr)==str:
        axis=axstr
    else:
        axis= veckeys[axstr]
    veckeys.remove(axis.lower())
    veckeys.append(axis.lower())
    datacoords = geod.dataloc
    xyzvecs = {l:sp.unique(datacoords[:,axdict[l]]) for l in veckeys}
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

    # get the data location, first check if the data can be just reshaped then do a
    # search

    sliceindx = slicenum==datacoords[:,axdict[axis]]

    datacoordred = datacoords[sliceindx]
    rstypes = ['C','F','A']
    nfounds = True
    M1dlfl = datacoordred[:,axdict[veckeys[0]]]
    M2dlfl = datacoordred[:,axdict[veckeys[1]]]
    for ir in rstypes:
        M1dl = sp.reshape(M1dlfl,M1.shape,order =ir)
        M2dl = sp.reshape(M2dlfl,M1.shape,order =ir)
        if sp.logical_and(sp.allclose(M1dl,M1),sp.allclose(M2dl,M2)):
            nfounds=False
            break
    if nfounds:
        dataout = geod.datareducelocation(new_coords,geod.coordnames,gkey)[:,time]
        dataout = sp.reshape(dataout,M1.shape)
    else:
        dataout = sp.reshape(geod.data[gkey][sliceindx,time],M1.shape,order=ir)

    title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()
    
    if vbounds is None:
        vbounds=[sp.nanmin(dataout),sp.nanmax(dataout)]
    if levels is None:
        levels=sp.linspace(vbounds[0],vbounds[1],5)
    if m is None:
        ploth = ax.contour(M1,M2,dataout,levels = levels,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)
        ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(),
                 xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
        if cbar:
            cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
        else:
            cbar2 = None
        ax.set_title(title)
        ax.set_xlabel(veckeys[0])
        ax.set_ylabel(veckeys[1])
    else:
        N1,N2 = m(M1,M2)
        ploth = ax.contour(N1,N2,dataout,levels = levels,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)

        if cbar:
            #cbar2 = m.colorbar(ploth,  format='%.0e')
            cbar2 = m.colorbar(ploth)
        else:
            cbar2 = None


    return(ploth,cbar2)

def scatterGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,
              ax=None,title='',cbar=True,err=.1,m=None):
    """ This will make a scatter plot given a GeoData object."""
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}# shows which row is this coordinate
        veckeys = ['long','lat','alt']# shows which is the x, y and z axes for plotting
    if type(axstr)==str:
        axis=axstr
    else:
        axis= veckeys[axstr]

    #determine the data name
    if gkey is None:
        gkey = geod.data.keys[0]
    geod=geod.timeslice(time)
    veckeys.remove(axis.lower())
    veckeys.append(axis.lower())
    datacoords = geod.dataloc
    xyzvecs = {l:sp.unique(datacoords[:,axdict[l]]) for l in veckeys}
    xyzvecsall = {l:datacoords[:,axdict[l]] for l in veckeys}
    if geod.issatellite():
        
        zdata = xyzvecsall[veckeys[2]]
        indxnum = np.abs(zdata-slicenum)<err
        xdata =xyzvecsall[veckeys[0]][indxnum]
        ydata =xyzvecsall[veckeys[1]][indxnum]
        dataout = geod.data[gkey][indxnum]
        title = insertinfo(title,gkey,geod.times[:,0].min(),geod.times[:,1].max())
    else:
        #make matrices
        xvec = xyzvecs[veckeys[0]]
        yvec = xyzvecs[veckeys[1]]
        M1,M2 = sp.meshgrid(xvec,yvec)
        slicevec = sp.unique(datacoords[:,axdict[axis]])
        min_idx = sp.argmin(sp.absolute(slicevec-slicenum))
        slicenum=slicevec[min_idx]
        rec_coords = {axdict[veckeys[0]]:M1.flatten(),axdict[veckeys[1]]:M2.flatten(),
                      axdict[axis]:slicenum*sp.ones(M2.size)}
        new_coords = sp.zeros((M1.size,3))
        xdata = M1.flatten()
        ydata= M2.flatten()

        #make coordinates
        for ckey in rec_coords.keys():
            new_coords[:,ckey] = rec_coords[ckey]


        # get the data location, first check if the data can be just reshaped then do a
        # search

        sliceindx = slicenum==datacoords[:,axdict[axis]]

        datacoordred = datacoords[sliceindx]
        rstypes = ['C','F','A']
        nfounds = True
        M1dlfl = datacoordred[:,axdict[veckeys[0]]]
        M2dlfl = datacoordred[:,axdict[veckeys[1]]]
        for ir in rstypes:
            M1dl = sp.reshape(M1dlfl,M1.shape,order =ir)
            M2dl = sp.reshape(M2dlfl,M1.shape,order =ir)
            if sp.logical_and(sp.allclose(M1dl,M1),sp.allclose(M2dl,M2)):
                nfounds=False
                break
        if nfounds:
            dataout = geod.datareducelocation(new_coords,geod.coordnames,gkey)[:,time]
            dataout = sp.reshape(dataout,M1.shape)
        else:
            dataout = sp.reshape(geod.data[gkey][sliceindx,time],M1.shape,order=ir)

        title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()
    if m is None:
        ploth = ax.scatter(xdata,ydata,c=dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)
        ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(),
                 xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
        if cbar:
            cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
        else:
            cbar2 = None
        ax.set_title(title)
        ax.set_xlabel(veckeys[0])
        ax.set_ylabel(veckeys[1])
    else:
        Xdata,Ydata = m(xdata,ydata)
        pdb.set_trace()
        ploth = m.scatter(Xdata,Ydata,c=dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)

        if cbar:
            cbar2 = m.colorbar(ploth)
        else:
            cbar2 = None

    return(ploth,cbar2)

def sliceGDsphere(geod,coordnames ='cartesian' ,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,ax=None,title='',cbar=True):

    assert geod.coordnames.lower() =='spherical'

    if coordnames.lower() in ['cartesian','enu','ecef']:
        veckeys = ['x','y','z']
    elif coordnames.lower() == 'wgs84':
        veckeys = ['lat','long','alt']

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

     #determine the data name
    if gkey is None:
        gkey = geod.data.keys[0]

    title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])

    xycoords = geod.__changecoords__(coordnames)

    xvec = xycoords[:,0]
    yvec = xycoords[:,1]
    curdata =geod.data[gkey][:,time]
    ploth = ax.tripcolor(xvec,yvec,curdata)
    if cbar:
        cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
    else:
        cbar2 = None

    ax.set_title(title)
    ax.set_xlabel(veckeys[0])
    ax.set_ylabel(veckeys[1])

    return(ploth,cbar2)

def plotbeamposfig(geod,height,coordnames,fig=None,ax=None,title=''):
    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    (beams,beaminds,beamnums) = uniquerows(geod.dataloc[:,1:])
    az = beams[:,0]
    el = beams[:,1]
    rho = height*np.tan(np.radians((90-el)))
    y = rho*np.cos(np.radians(az))
    x = rho*np.sin(np.radians(az))

    ploth = ax.scatter(x,y)
    return(ploth)

def rangevstime(geod,beam,vbounds=(None,None),gkey = None,cmap=None,fig=None,ax=None,
                title='',cbar=True,tbounds=(None,None),ic=True,ir=True,it=True):
    """ This method will create a color graph of range vs time for data in spherical coordinates"""
    assert geod.coordnames.lower() =='spherical', 'I expect speherical coordinate data'

    if (ax is None) and (fig is None):
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    if gkey is None:
        gkey = geod.data.keys[0]
#%% get unique ranges for plot limits, note beamid is not part of class.
    match = np.isclose(geod.dataloc[:,1:],beam,atol=1e-2).all(axis=1) #TODO what should tolerance be for Sondrestrom mechanical dish
    if (~match).all(): #couldn't find this beam
        logging.error('beam az,el {} not found'.format(beam))
        return

    if not title:
        title = gkey

    dataout = geod.data[gkey][match]
    rngval =  geod.dataloc[match,0]
    t = np.asarray(list(map(dt.datetime.utcfromtimestamp, geod.times[:,0])))
#%% time limits of display
    ploth = ax.pcolormesh(t,rngval,dataout,
                          vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)

    if cbar:
        fig.colorbar(ploth, ax=ax, format=sfmt)

    if it:
        ax.set_title(title)
    if ic:
        ax.set_ylabel('az,el = {} \n slant range [km]'.format(beam))
    if ir:
        ax.set_xlabel('UTC')

    ttxt = tbounds[0].strftime('%Y-%m-%d') if tbounds[0] else t[0].strftime('%Y-%m-%d')
    fig.suptitle(ttxt,fontsize='xx-large')

    ax.autoscale(axis='y',tight=True) #fills axis
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim(tbounds)
    fig.autofmt_xdate()

def uniquerows(a):
    b=np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    (rowsinds,rownums) = np.unique(b,return_index=True, return_inverse=True)[1:]
    rows = a[rowsinds]
    return (rows,rowsinds,rownums)

def plotbeamposGD(geod,title='Beam Positions',minel=30,elstep=10):
    assert geod.coordnames.lower() =='spherical'

    (azvec,elvec) = (geod.dataloc[:,1],geod.dataloc[:,2])

    polarplot(azvec,elvec,markerarea=70,title=title,minel=minel,elstep=elstep)

def make_polax(zenith):
    """ OBSOLETE
    This makes the polar axes for the beams"""
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

def polarplot(az,el,markerarea=500,title=None,minel=30,elstep=10):
    """
    plots hollow circles at az,el coordinates, with area quantitatively defined
    Michael Hirsch from satkml
    """
    az = np.radians(np.asarray(az).astype(float))
    el = 90-np.asarray(el).astype(float)

    ax=plt.figure().gca(polar=True)

    ax.set_theta_zero_location('N')
#    ax.set_rmax(90-minel)
    ax.set_theta_direction(-1)

    ax.scatter(x=az, y=el, marker='o',facecolors='none',edgecolor='red',s=markerarea)

    yt = np.arange(0., 90.-minel+elstep, elstep)
    ax.set_yticks(yt)
    ylabel = (yt[::-1]+minel).astype(int).astype(str)
    ax.set_yticklabels(ylabel)

    ax.set_title(title)

#quiver() creates quiver plots with contours from GeoData objects
#arrowscale is the scale of the quiver plot vector arrows

def quiverGD(geod,axstr,slicenum,arrowscale,vbounds=None,time = 0,gkey = None,cmap='jet', fig=None,ax=None,title='',cbar=True,m=None):
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}# shows which row is this coordinate
        veckeys = ['long','lat','alt']# shows which is the x, y and z axes for plotting
    if type(axstr)==str:
        axis=axstr
    else:
        axis= veckeys[axstr]
    veckeys.remove(axis.lower())
    veckeys.append(axis.lower())
    datacoords = geod.dataloc
    xyzvecs = {l:sp.unique(datacoords[:,axdict[l]]) for l in veckeys}
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
        gkey = geod.data.keys()[0]

    # get the data location, first check if the data can be just reshaped then do a
    # search

    sliceindx = slicenum==datacoords[:,axdict[axis]]

    datacoordred = datacoords[sliceindx]
    rstypes = ['C','F','A']
    nfounds = True
    M1dlfl = datacoordred[:,axdict[veckeys[0]]]
    M2dlfl = datacoordred[:,axdict[veckeys[1]]]
    for ir in rstypes:
        M1dl = sp.reshape(M1dlfl,M1.shape,order =ir)
        M2dl = sp.reshape(M2dlfl,M1.shape,order =ir)
        if sp.logical_and(sp.allclose(M1dl,M1),sp.allclose(M2dl,M2)):
            nfounds=False
            break
    if nfounds:
        
        dx = geod.datareducelocation(new_coords,geod.coordnames,gkey[0])[:,time]
        dy = geod.datareducelocation(new_coords,geod.coordnames,gkey[1])[:,time]
        dx = sp.reshape(dx,M1.shape)
        dy = sp.reshape(dy,M1.shape)
    else:
        dx = sp.reshape(geod.data[gkey[0]][sliceindx,time],M1.shape,order=ir)
        dy = sp.reshape(geod.data[gkey[1]][sliceindx,time],M1.shape,order=ir)

    
    title = insertinfo(title,gkey[0],geod.times[time,0],geod.times[time,1])

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    if m is None:
        
        
        quiv = ax.quiver(M1,M2,dx,dy,scale=arrowscale)
            
        ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(),
                 xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
        
        ax.set_title(title)
        ax.set_xlabel(veckeys[0])
        ax.set_ylabel(veckeys[1])
    else:
        N1,N2 = m(M1,M2)
        
        quiv = ax.quiver(M1,M2,dx,dy,scale=arrowscale)


    return(quiv)

def insertinfo(strin,key='',posix=None,posixend = None):

    listin = isinstance(strin,list)
    if listin:
        stroutall = []
    else:
        strin=[strin]

    for k in range(len(strin)):

        strout = strin[k].replace('$k',key)
        if posix is None:
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
                '$tmdy',#UT month/day/year
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
                time.strftime('%m/%d/%Y',curdt),
                time.strftime('%m/%d %H:%M',curdt)+' UT']
            for imark in range(len(markers)):
                strout=strout.replace(markers[imark],datestrcell[imark]);

        if listin:
            stroutall[k] = strout
        else:
            stroutall = strout
    return stroutall

def plotazelscale(opt,az=None,el=None):
    """
    diagnostic: plots az/el map over test image
    Michael Hirsch
    """
    if isinstance(opt,GeoData):
        img = opt.data['optical'][0,...]
        az = opt.dataloc[:,1].reshape(img.shape)
        el = opt.dataloc[:,2].reshape(img.shape)
    elif isinstance(opt,np.ndarray):
        img = opt
    else:
        raise NotImplementedError('not sure what your opt array {} is'.format(type(opt)))

    assert img.ndim==2, 'just one image please'
    assert img.shape==az.shape==el.shape,'do you need to reshape your az/el into 2-D like image?'

    fg,ax = plt.subplots(1,2,figsize=(12,6))
    for a,q,t in zip(ax,(az,el),('azimuth','elevation')):
        a.imshow(img,origin='lower',interpolation='none',cmap='gray')
        c=a.contour(q)
        a.clabel(c, inline=1,fmt='%0.1f')
        a.set_title(t)
        a.grid(False)

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
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib import ticker
try:
    from mayavi import mlab
except Exception as e:
    pass
#
from .CoordTransforms import angles2xy
#
if False:
    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    except Exception as e:
        logging.info('Latex install not complete, falling back to basic fonts.  sudo apt-get install dvipng')
#
try:
    import seaborn as sns
    sns.color_palette(sns.color_palette("cubehelix"))
    sns.set(context='poster', style='whitegrid')
    sns.set(rc={'image.cmap': 'cubehelix_r'}) #for contour
except Exception as e:
    logging.info('could not import seaborn  {}'.format(e))
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
    except Exception as e:
        logging.warning('skipping instrument   {}'.format(e))
#%% isr
    g = geodatalist[1]
    try:
        key['isr'] = list(g.data.keys()) #list necessary for Python3
        G = g.timeslice(picktimeind)
        G.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
        interpData = G.data[key['isr'][0]]
        isr = interpData[:,0].reshape(x.shape)
    except Exception as e:
        logging.warning('skipping instrument   {}'.format(e))

    return opt,isr,extent,key,x,y
#%%
def alt_slice_overlay(geodatalist, altlist, xyvecs, vbounds, title, axis=None,picktimeind=[1,2]):
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
        logging.info('problem plotting instrument  {}'.format(e))
#%%
    try:
        top = ax.imshow(isr, alpha=0.4, extent=extent, origin='lower',interpolation='none',
                        vmin=vbounds[1][0],vmax=vbounds[1][1])
        c = fg.colorbar(top,ax=ax)
        c.set_label(key['isr'][0])
    except Exception as e:
        logging.info('problem plotting instrument  {}'.format(e))

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
        logging.info('problem plotting instrument  {}'.format(e))

    try:
        top = ax.contour(x,y, isr,extent=extent, origin='lower', interpolation='none',
                         vmin=vbounds[1][0],vmax=vbounds[1][1])
        #clabel(top,inline=1,fontsize=10, fmt='%1.0e')
        cbar2 = fg.colorbar(top, format='%.0e',ax=ax)
        cbar2.set_label(key['isr'][0])
    except Exception as e:
        logging.info('problem plotting instrument  {}'.format(e))

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

def slice2DGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,ax=None,title='',cbar=True):

    #xyzvecs is the area that the data covers.
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}
        veckeys = ['lat','long','alt']
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
    dataout = geod.datareducelocation(new_coords,geod.coordnames,gkey)[:,time]


    title = insertinfo(title,gkey,geod.times[time,0],geod.times[time,1])
    dataout = sp.reshape(dataout,M1.shape)

    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    ploth = ax.pcolor(M1,M2,dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)
    ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(), xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
    if cbar:
        cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
    else:
        cbar2 = None
    ax.set_title(title)
    ax.set_xlabel(veckeys[0])
    ax.set_ylabel(veckeys[1])

    return(ploth,cbar2)

def contourGD(geod,axstr,slicenum,vbounds=None,time = 0,gkey = None,cmap='jet',fig=None,ax=None,title='',cbar=True):
    poscoords = ['cartesian','wgs84','enu','ecef']
    assert geod.coordnames.lower() in poscoords

    if geod.coordnames.lower() in ['cartesian','enu','ecef']:
        axdict = {'x':0,'y':1,'z':2}
        veckeys = ['x','y','z']
    elif geod.coordnames.lower() == 'wgs84':
        axdict = {'lat':0,'long':1,'alt':2}
        veckeys = ['lat','long','alt']
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

    ploth = ax.contour(M1,M2,dataout,vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)
    ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(), xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
    if cbar:
        cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
    else:
        cbar2 = None
    ax.set_title(title)
    ax.set_xlabel(veckeys[0])
    ax.set_ylabel(veckeys[1])

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
    d2r = sp.pi/180.
    if (ax is None) and (fig is None):
        fig = plt.figure(facecolor='white')
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    (beams,beaminds,beamnums) = uniquerows(geod.dataloc[:,1:])
    az = beams[:,0]
    el = beams[:,1]
    rho = height*sp.tan((90-el)*d2r)
    y = rho*sp.cos(az*d2r)
    x = rho*sp.sin(az*d2r)

    ploth = ax.scatter(x,y)
    return(ploth)

def rangevstime(geod,beam,vbounds=(None,None),gkey = None,cmap=None,fig=None,ax=None,
                title='',cbar = True,tbounds=(None,None),ind=None):
    """ This method will create a color graph of range vs time for data in spherical coordinates"""
    assert geod.coordnames.lower() =='spherical'

    usingsubplot=False
    if (ax is None) and (fig is None):
        fig = plt.figure(figsize=(12,8))
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()
    else:
        usingsubplot=True #assuming sharey

    if gkey is None:
        gkey = geod.data.keys[0]
#%% get unique ranges for plot limits, note beamid is not part of class.
    match = np.isclose(geod.dataloc[:,1:],beam).all(axis=1)

    title = insertinfo(title,gkey)

    dataout = geod.data[gkey][match]
    rngval =  geod.dataloc[match,0]
    t = np.asarray(list(map(dt.datetime.utcfromtimestamp, geod.times[:,0])))
#%% time limits of display
    ploth = ax.pcolormesh(t,rngval,dataout,
                          vmin=vbounds[0], vmax=vbounds[1],cmap = cmap)

    if cbar:
        fig.colorbar(ploth, ax=ax, format=sfmt)

    ax.set_title(title)
    ax.set_xlabel('UTC')

    if usingsubplot and ind==0:
        ax.set_ylabel('slant range [km]')

    ax.autoscale(axis='y',tight=True) #fills axis
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim(tbounds)
    fig.autofmt_xdate()

def uniquerows(a):
    b=np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    (rowsinds,rownums) = np.unique(b,return_index=True, return_inverse=True)[1:]
    rows = a[rowsinds]
    return (rows,rowsinds,rownums)

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

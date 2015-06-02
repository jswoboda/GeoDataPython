# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: Anna Stuhlmacher

plotting
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as spinterp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from matplotlib import ticker
from mayavi import mlab

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
    x,y = sp.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    np.ndarray.flatten(x)
    np.ndarray.flatten(y)
    np.ndarray.flatten(z)
    new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    key0 = geodatalist[0].data.keys()
    key1 =  geodatalist[1].data.keys()

    gd2 = geodatalist[1].timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method='linear', fill_value=np.nan)
    interpData = gd2.data[key1[0]]
    risr = interpData[:,0].reshape(x.shape)

    gd3 = geodatalist[0].timeslice([1,2])
    gd3.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd3.data[key0[0]]
    omti = interpData[:,0].reshape(x.shape)

    if axis == None:
        plt.figure(facecolor='white')
        bottom = imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        cbar1 = plt.colorbar(bottom)
        cbar1.set_label(key0[0])
        hold(True)
        top = imshow(risr, cmap=cm.jet, alpha=0.4, extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        cbar2 = plt.colorbar(top)
        cbar2.set_label(key1[0])
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        show()
    else:
        axis.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        hold(True)
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
    x,y = sp.meshgrid(xvec, yvec)
    z = np.ones(x.shape)*altlist
    np.ndarray.flatten(x)
    np.ndarray.flatten(y)
    np.ndarray.flatten(z)
    new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
    extent=[xvec.min(),xvec.max(),yvec.min(),yvec.max()]

    key0 = geodatalist[0].data.keys()
    key1 =  geodatalist[1].data.keys()

    gd2 = geodatalist[1].timeslice([1,2])
    gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd2.data[key1[0]]
    risr = interpData[:,0].reshape(x.shape)

    gd3 = geodatalist[0].timeslice([1,2])
    gd3.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
    interpData = gd3.data[key0[0]]
    omti = interpData[:,0].reshape(x.shape)

    if axis == None:
        plt.figure(facecolor='white')
        bottom = imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        cbar1 = plt.colorbar(bottom, orientation='horizontal')
        cbar1.set_label(key0[0])
        hold(True)
        top = contour(x,y, risr, cmap=cm.jet,extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        #clabel(top,inline=1,fontsize=10, fmt='%1.0e')
        cbar2 = plt.colorbar(top, format='%.0e')
        cbar2.set_label(key1[0])
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        show()
    else:
        axis.imshow(omti, cmap=cm.gray, extent=extent, origin='lower', vmin=vbounds[0][0],vmax=vbounds[0][1])
        hold(True)
        axis.contour(x,y, risr, cmap=cm.jet,extent=extent, origin='lower', vmin=vbounds[1][0],vmax=vbounds[1][1])
        return axis


def plot3D(geodata, altlist, xyvecs, vbounds, title, time = 0,gkey = None, ax=None,fig=None):
    """
    Inputs:
    geodata - A geodata object that will be plotted in 3D
    altlist - A list of the altitudes that RISR parameter slices will be taken at
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for the geodata objec's parameters. ie, vbounds=[500,2000]
    title - A string that holds for the overall image
    ax - A handle for an axis that this will be plotted on.

    Returns an 3D image of the different altitude slices for the geodata object parameter that is passed in.
    """
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    x,y = sp.meshgrid(xvec, yvec)
    x_tot, y_tot, z_tot, p_tot = (np.ones(x.shape) for i in range(4))
    np.ndarray.flatten(x)
    np.ndarray.flatten(y)
    key = geodata.data.keys()
    xlen = x.shape[0]
    if gkey is None:
        gkey = key[0]
    xydict = {0:xyvecs[0],1:xyvecs[1]}
    for alt in altlist:
         z = np.ones(x.shape)*alt
         np.ndarray.flatten(z)
         new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
         gd2 = geodata.copy().timeslice([time])
         zloc = sp.argmin(sp.absolute(geodata.dataloc[:,2]-alt))
         zval = geodata.dataloc[zloc,2]
         xydict[2]=zval
         gdcheck = gd2.getdatalocationslice(xydict,newcoordname = 'Cartesian',copyinst=False)
         if not gdcheck:
             gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)

         xloc = sp.unique(gd2.dataloc[:,0],return_inverse = True)[1]
         yloc = sp.unique(gd2.dataloc[:,1],return_inverse = True)[1]
         p1 = sp.zeros(x.shape)
         interpData = gd2.data[gkey]

         p1[yloc,xloc] = interpData[:,time]
         x_tot = np.concatenate((x_tot,x), axis=0)
         y_tot = np.concatenate((y_tot,y), axis=0)
         z_tot = np.concatenate((z_tot,z), axis=0)
         p_tot = np.concatenate((p_tot,p1), axis=0)

    if ax is None:
        fig = plt.figure(facecolor='white')
        ax = fig.gca(projection='3d')
    N_tot = p_tot/np.nanmax(p_tot) #normalize (0...1)

    surf = ax.plot_surface(x_tot[xlen:][:], y_tot[xlen:][:], z_tot[xlen:][:],
                           rstride=1, cstride=1, facecolors=cm.jet(N_tot[xlen:][:]),
                           linewidth=0, antialiased=False,vmin=vbounds[0],
                           vmax=vbounds[1], shade=False)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim3d(min(altlist)-50, max(altlist)+50)
    #creating colorbar
    #axes dimensions in fractions of figure image [left, bottom, width, height]
    ax_color = fig.add_axes([0.9, 0.05, 0.03, 0.80])
    #choosing color map scheme and defining the bounds of the normalized bar
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=vbounds[0], vmax=vbounds[1])
    cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cmap,norm=norm, orientation='vertical')
    cb1.set_label(gkey)
    #plt.show()
def plot3D2(geodata, altlist, xyvecs, vbounds, title, time = 0,gkey = None, ax=None,fig=None):
    """
    Inputs:
    geodata - A geodata object that will be plotted in 3D
    altlist - A list of the altitudes that RISR parameter slices will be taken at
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for the geodata objec's parameters. ie, vbounds=[500,2000]
    title - A string that holds for the overall image
    ax - A handle for an axis that this will be plotted on.

    Returns an 3D image of the different altitude slices for the geodata object parameter that is passed in.
    """
    xvec = xyvecs[0]
    yvec = xyvecs[1]
    zvec = sp.array(altlist)
    x,y,z = sp.meshgrid(xvec, yvec,zvec)

    key = geodata.data.keys()
    xlen = x.shape[0]
    if gkey is None:
        gkey = key[0]
    xydict = {0:xyvecs[0],1:xyvecs[1],2:zvec}
    new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
    gd2 = geodata.copy().timeslice([time])
    gdcheck = gd2.datalocationslice(xydict,newcoordname = 'Cartesian',copyinst=False)
    if not gdcheck:
        gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)

    for iz in range(len(altlist)):
        xt,yt,zt = [sp.reshape(sp.transpose(gd2.dataloc)[idim,iz::len(altlist)],x.shape[:2]) for idim in range(3)]
        pt =sp.reshape( gd2.data[gkey][iz::len(altlist),time],x.shape[:2])
        if iz==0:
            x_tot = xt
            y_tot= yt
            z_tot=zt
            p_tot=pt
        else:
            x_tot = np.concatenate((x_tot,xt), axis=0)
            y_tot = np.concatenate((y_tot,yt), axis=0)
            z_tot = np.concatenate((z_tot,zt), axis=0)
            p_tot = np.concatenate((p_tot,pt), axis=0)

    if ax is None:
        fig = plt.figure(facecolor='white')
        ax = fig.gca(projection='3d')
    N_tot = p_tot/np.nanmax(p_tot) #normalize (0...1)

    surfs = ax.plot_surface(x_tot[xlen:][:], y_tot[xlen:][:], z_tot[xlen:][:],
                           rstride=1, cstride=1, facecolors=cm.jet(N_tot[xlen:][:]),
                           linewidth=0, antialiased=False,vmin=vbounds[0],
                           vmax=vbounds[1], shade=False)
    plt.title(title)
    plt.xlabel('x in km')
    plt.ylabel('y in km')
    ax.set_zlabel('z in km')
    ax.set_zlim3d(min(altlist)-50, max(altlist)+50)
    #creating colorbar
    #axes dimensions in fractions of figure image [left, bottom, width, height]
    ax_color = fig.add_axes([0.9, 0.05, 0.03, 0.80])
    #choosing color map scheme and defining the bounds of the normalized bar
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=vbounds[0], vmax=vbounds[1])
    cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cmap,norm=norm, orientation='vertical')
    cb1.set_label(gkey)

def plot3Dslice(geodata,surfs,vbounds, titlestr='', time = 0,gkey = None,cmap='jet', ax=None,fig=None,method='linear',fill_value=np.nan):
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
    if islists or onedim:
        p = np.reshape(porig,matshape,order= ord )
        xslices = surfs[0]
        for isur in xslices:
            indx = sp.argmin(sp.absolute(isur-xvec))
            xtmp = x[:,indx]
            ytmp = y[:,indx]
            ztmp = z[:,indx]
            ptmp = p[:,indx]
            mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap)

        yslices = surfs[1]
        for isur in yslices:
            indx = sp.argmin(sp.absolute(isur-yvec))
            xtmp = x[indx]
            ytmp = y[indx]
            ztmp = z[indx]
            ptmp = p[indx]
            mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap)

        zslices = surfs[2]
        for isur in zslices:
            indx = sp.argmin(sp.absolute(isur-zvec))
            xtmp = x[:,:,indx]
            ytmp = y[:,:,indx]
            ztmp = z[:,:,indx]
            ptmp = p[:,:,indx]
            mlab.mesh(xtmp,ytmp,ztmp,scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap)
    else:
        # For a general surface.
        xtmp,ytmp,ztmp = surfs[:]
        gooddata = ~np.isnan(porig)
        curparam = porig[gooddata]
        curlocs = datalocs[gooddata]
        new_coords = np.column_stack((xtmp.flatten(),ytmp.flatten(),ztmp.flatten()))
        ptmp = spinterp.griddata(curlocs,curparam,new_coords,method,fill_value)
        mlab.mesh(surfs[0],surfs[1],surfs[2],scalars=ptmp,vmin=vbounds[0],vmax=vbounds[1],colormap=cmap)
    mlab.colorbar(title=gkey, orientation='vertical')
    mlab.title(titlestr,color=(0,0,0))
    mlab.outline(color=(0,0,0))
    mlab.axes(color=(0,0,0),x_axis_visibility=True,xlabel = 'x in km',y_axis_visibility=True,
              ylabel = 'y in km',z_axis_visibility=True,zlabel = 'z in km')

    mlab.orientation_axes(xlabel = 'x in km',ylabel = 'y in km',zlabel = 'z in km')

def plotbeamposGD(geod,fig=None,ax=None,title=''):
        assert geod.coordnames.lower() =='spherical'

def slice2D(geod,xyzvecs,slicenum,vbounds,axis='z',time = 0,gkey = None,fig=None,ax=None,title=''):
    axdict = {'x':0,'y':1,'z':2}
    veckeys = xyzvecs.keys()
    if axis in veckeys: veckeys.remove(axis)
    veckeys.sort()
    M1,M2 = sp.meshgrid(xyzvecs[veckeys[0]],xyzvecs[veckeys[1]])
    rec_coords = {axdict[veckeys[0]]:M1.flatten(),axdict[veckeys[1]]:M2.flatten(),
                  axdict[axis]:slicenum*sp.ones(M2.size)}
    new_coords = sp.zeros((M1.size,3))
    for ckey in rec_coords.keys(): new_coords[:,ckey] = rec_coords[ckey]

    if gkey is None: gkey = geod.data.keys[0]

    dataout = geod.getdatalocationslice(rec_coords,gkey,newcoordname = 'Cartesian')
    if dataout:
        dataout = dataout[:,time]
    if not dataout:
        gd2 = geod.copy().timeslice([time])
        gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
        dataout = gd2.data[gkey]
    dataout = sp.reshape(dataout,M1.shape)

    if ax is None:
        fig = plt.figure(facecolor='white')
        ax = fig.gca(projection='3d')

    ploth = ax.pcolor(M1,M2,dataout,vmin=vbounds[0], vmax=vbounds[1])
    ax.axis([xyzvecs[veckeys[0]].min(), xyzvecs[veckeys[0]].max(), xyzvecs[veckeys[1]].min(), xyzvecs[veckeys[1]].max()])
    cbar2 = plt.colorbar(ploth, ax=ax, format='%.0e')
    cbar2.set_label(gkey)
    ax.set_title(title)
    ax.set_xlabel(veckeys[0])
    ax.set_ylabel(veckeys[1])

    return()


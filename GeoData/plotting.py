# -*- coding: utf-8 -*-
"""
Created on Fri Jan 02 09:38:14 2015

@author: Anna Stuhlmacher

plotting
"""

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from matplotlib import ticker
   
        
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
    

def plot3D (geodata, altlist, xyvecs, vbounds, title, axis=None):
    """
    Inputs:
    geodata - A geodata object that will be plotted in 3D
    altlist - A list of the altitudes that RISR parameter slices will be taken at
    xyvecs- A list of x and y numpy arrays that have the x and y coordinates that the data will be interpolated over. ie, xyvecs=[np.linspace(-100.0,500.0),np.linspace(0.0,600.0)]
    vbounds = a list of bounds for the geodata objec's parameters. ie, vbounds=[500,2000]
    title - A string that holds for the overall image
    
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

    for alt in altlist:
         z = np.ones(x.shape)*alt
         np.ndarray.flatten(z)
         new_coords = np.column_stack((x.flatten(),y.flatten(),z.flatten()))
         gd2 = geodata.copy().timeslice([1,2])
         gd2.interpolate(new_coords, newcoordname='Cartesian', method='nearest', fill_value=np.nan)
         interpData = gd2.data[key[0]]
         p1 = interpData[:,0].reshape(x.shape)
         x_tot = np.concatenate((x_tot,x), axis=0)
         y_tot = np.concatenate((y_tot,y), axis=0)
         z_tot = np.concatenate((z_tot,z), axis=0)
         p_tot = np.concatenate((p_tot,p1), axis=0)
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    N_tot = p_tot/np.nanmax(p_tot) #normalize (0...1)
    surf = ax.plot_surface(x_tot[xlen:][:], y_tot[xlen:][:], z_tot[xlen:][:], rstride=1, cstride=1, facecolors=cm.jet(N_tot[xlen:][:]), linewidth=0, antialiased=False, vmin=vbounds[0], vmax=vbounds[1], shade=False) 
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim3d(250, 550)
    #creating colorbar
    #axes dimensions in fractions of figure image [left, bottom, width, height]
    ax_color = fig.add_axes([0.9, 0.05, 0.03, 0.80])
    #choosing color map scheme and defining the bounds of the normalized bar
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=vbounds[0], vmax=vbounds[1])
    cb1 = mpl.colorbar.ColorbarBase(ax_color, cmap=cmap,norm=norm, orientation='vertical')
    cb1.set_label(key[0])
    plt.show()

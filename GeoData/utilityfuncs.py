#!/usr/bin/env python
"""
Created on Thu Sep 11 15:29:27 2014

@author: John Swoboda
"""
import pdb
import numpy as np
from tables import *
import os
import time
import posixpath
import sys
from copy import copy
import scipy as sp
import scipy.interpolate as spinterp
import CoordTransforms as CT


VARNAMES = ['data','coordnames','dataloc','sensorloc','times']

def readMad_hdf5 (filename, paramstr): #timelims=None
    """@author: Anna Stuhlmacher
    madgrigal h5 read in function for the python implementation of GeoData for Madrigal Sondrestrom data
    Input:
    filename path to hdf5 file
    list of parameters to look at written as strings
    Returns:
    dictionary with keys are the Madrigal parameter string, the value is an array
    rows are unique data locations (data_loc) = (rng, azm, el1)
    columns are unique times
    """
    #open hdf5 file
    files = openFile(filename, mode = "r", title = 'Sondrestrom1')
    all_data = files.getNode('/Data/Table Layout').read()
    sensor_data = files.getNode('/Metadata/Experiment Parameters').read()
    files.close()

    instrument = sensor_data[0][1] #instrument type string
    if "Sondrestrom" in instrument:
        radar = 1
        print("Sondrestrom data")
    elif "Poker Flat" in instrument:
        radar = 2
        print("PFISR data")
    elif "Resolute Bay" in instrument:
        radar = 3
        print ("RISR data")
    else:
        print("Error: Radar type not supported by program in this version.")
        exit()

    # get the data location (range, el1, azm)
    all_loc = []
    if radar == 1:
        angle1 = 'elm'
        rng = all_data['gdalt']
    elif radar == 2:
        angle1 = 'elm'
        rng = all_data['range']
    elif radar ==3:
        angle1 = 'elm'
        rng = all_data['range']

    try:
        el = all_data[angle1]
    except ValueError:
        el = np.nan * np.ones(len(list(rng)))

    try:
        azm = all_data['azm']
    except ValueError:
        azm = np.nan * np.ones(len(list(rng)))
    # take out nans
    nan_ar = np.isnan(rng)|np.isnan(el)|np.isnan(azm)
    notnan = np.logical_not(nan_ar)
    for i in range(len(rng)):
        if notnan[i]:
            all_loc.append([rng[i],azm[i],el[i]])

    #create list of unique data location lists
    dataloc = [list(y) for y in set([tuple(x) for x in all_loc])]
    all_times = []
    times1 = all_data['ut1_unix'][notnan]
    times2 = all_data['ut2_unix'][notnan]
    for i in range(len(times1)):
        all_times.append([times1[i], times2[i]])
    uniq_times = sorted(set(tuple(x) for x in all_times))

    #initialize and fill data dictionary with parameter arrays
    data = {}
    maxcols = len(uniq_times)
    maxrows = len(dataloc)
    for p in paramstr:
        if not p in all_data.dtype.names:
            print 'Warning: ' + p + ' is not a valid parameter name.'
            continue
        tempdata = all_data[p][notnan] #list of parameter pulled from all_data
        temparray = np.empty([maxrows,maxcols]) #converting the tempdata list into array form

        for t in range(len(tempdata)):
            #row
            row = dataloc.index(all_loc[t])
            #column-time
            col = uniq_times.index(tuple(all_times[t]))
            temparray[row][col] = tempdata[t]

        data[p]=temparray

    #get the sensor location (lat, long, rng)
    lat = sensor_data[7][1]
    lon = sensor_data[8][1]
    sensor_alt = sensor_data[9][1]
    sensorloc = np.array([lat,lon,sensor_alt], dtype='f')
    coordnames = 'Spherical'

    return (data,coordnames,np.array(dataloc, dtype='f'),sensorloc,np.asarray(uniq_times, dtype='f'))

def read_h5_main(filename):
    ''' Read in the structured h5 file.'''
    h5file=openFile(filename)
    output={}
    # Read in all of the info from the h5 file and put it in a dictionary.
    for group in h5file.walkGroups(posixpath.sep):
        output[group._v_pathname]={}
        for array in h5file.listNodes(group, classname = 'Array'):
            output[group._v_pathname][array.name]=array.read()
    h5file.close()
    #pdb.set_trace()
    # find the base paths which could be dictionaries or the base directory
    outarr = [pathparts(ipath) for ipath in output.keys() if len(pathparts(ipath))>0]
    outlist = []
    basekeys  = output[posixpath.sep].keys()
    # Determine assign the entries to each entry in the list of variables.
    # Have to do this in order because of the input being a list instead of a dictionary
    for ivar in VARNAMES:
        dictout = False
        #dictionary
        for npath,ipath in enumerate(outarr):
            if ivar==ipath[0]:
                outlist.append(output[output.keys()[npath]])
                dictout=True
                break
        if dictout:
            continue
        # for non-dicitonary
        for ikeys in basekeys:
            if ikeys==ivar:
                # Have to check for MATLAB type strings, for some reason python does not like to register them as strings
                curdata = output[posixpath.sep][ikeys]
                if type(curdata)==np.ndarray:
                    if curdata.dtype.kind=='S':
                        curdata=str(curdata)
                outlist.append(curdata)

    return tuple(outlist)

def pathparts(path):
    ''' '''
    components = []
    while True:
        (path,tail) = posixpath.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)

def readOMTI(filename, paramstr):
    outlist = read_h5_main(filename)
    optical = outlist[0]
    enu = outlist[2]
    dataloc = CT.enu2cartisian(enu)
    coordnames = 'Cartesian'
    sensorloc = outlist[3]
    times = outlist[4]

    return (optical, coordnames, np.array(dataloc, dtype='f'), sensorloc, np.asarray(times,dtype='f'))

def readIono(iono):
    """ This function will break in stances of the IonoContainer class"""
    pnames = iono.Param_Names
    Param_List = iono.Param_List
    (nloc,nt) = Param_List.shape[:2]
    if type(pnames) == sp.ndarray:
        if pnames.ndim>1:
            ionkeys = pnames.flatten()
            Param_List = sp.reshape(Param_List,(nloc,nt,len(ionkeys)))
    paramdict = {ikeys:Param_List[:,:,ikeyn] for ikeyn, ikeys in enumerate(ionkeys)}
    if iono.Coord_Vecs == ['r','theta','phi']:
        coordnames = 'Sphereical'
        coords = iono.Sphere_Coords
    elif iono.Coord_Vecs == ['x','y','z']:
        coordnames = 'Cartesian'
        coords = iono.Cart_Coords
    return (paramdict,coordnames,coords,sp.array(iono.Sensor_loc),iono.Time_Vector)

#data, coordnames, dataloc, sensorloc, times = readMad_hdf5('/Users/anna/Research/Ionosphere/2008WorldDaysPDB/son081001g.001.hdf5', ['ti', 'dti', 'nel'])

#!/usr/bin/env python
"""
Created on Thu Sep 11 15:29:27 2014

@author: John Swoboda
"""

import numpy as np
from tables import *


def readMad_hdf5 (filename, paramstr): #timelims=None
    """@author: Anna Stuhlmacher
    madgrigal h5 read in function for the python implementation of GeoData for Madrigal Sondrestrom data
    Input:
    filename path to hdf5 file
    list of parameters to look at written as strings
    Returns:
    dictionary with keys are the Madrigal parameter string, the value is an array
    rows are unique data locations (data_loc) = (alt, el1, azm)
    columns are unique times
    """
    #open hdf5 file    
    files = openFile(filename, mode = "r", title = 'Sondrestrom1')
    all_data = files.getNode('/Data/Table Layout').read()
    sensor_data = files.getNode('/Metadata/Experiment Parameters').read()
    files.close() 
    
    # get the data location (range, el1, azm)
    all_loc = []
    alt = all_data['gdalt']  
    try:
        el1 = all_data['el1']
    except ValueError:
        el1 = ['NaN']*len(list(alt))
    try:       
        azm = all_data['azm']
    except ValueError:        
        azm = ['NaN']*len(list(alt))   
    for i in range(len(alt)):
        all_loc.append([alt[i], el1[i], azm[i]])

    #create list of unique data location lists
    dataloc = [list(y) for y in set([tuple(x) for x in all_loc])]
    
    #create N x 1 time array
    times = all_data['ut1_unix']
    uniq_times = sorted(set(times))
    times = np.asarray(times)
        
    #initialize and fill data dictionary with parameter arrays
    data = {}
    maxcols = len(uniq_times)
    maxrows = len(dataloc)
    for p in paramstr:
        if not p in all_data.dtype.names:
            print 'Warning: ' + p + ' is not a valid parameter name.'
            continue
        tempdata = all_data[p] #list of parameter pulled from all_data
        temparray = np.empty([maxrows,maxcols]) #converting the tempdata list into array form
        
        for t in range(len(tempdata)):
            #row
            row = dataloc.index(all_loc[t])
            #column-time
            col = uniq_times.index(times[t])
            temparray[row][col] = tempdata[t]

        data[p]=temparray
 
    #get the sensor location (lat, long, alt)
    lat = sensor_data[7][1]
    lon = sensor_data[8][1]
    sensor_alt = sensor_data[9][1]
    sensorloc = np.array([lat,lon,sensor_alt])
    #sensor_type = "Sondrestrom"
    coordnames = 'Spherical'
    
    return (data,coordnames,dataloc,sensorloc,times)
    
    
#data, coordnames, dataloc, sensorloc, times, sensor_type = readMad_hdf5('/Users/anna/Research/Ionosphere/2008WorldDaysPDB/son081001g.001.hdf5', ['ti', 'dti', 'nel'])

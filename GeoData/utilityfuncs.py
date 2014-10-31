#!/usr/bin/env python
"""
Created on Thu Sep 11 15:29:27 2014

@author: John Swoboda
"""
import pdb
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
        alt = all_data['gdalt'] 
    elif radar == 2:
        angle1 = 'elm'
        alt = all_data['range'] 
    elif radar ==3:
        angle1 = 'elm'
        alt = all_data['range'] 
        
    try:
        el = all_data[angle1]
    except ValueError:
        el = np.nan * np.ones(len(list(alt)))
        
    try:       
        azm = all_data['azm']
    except ValueError:        
        azm = np.nan * np.ones(len(list(alt)))
    # take out nans
    nan_ar = np.isnan(alt)|np.isnan(el)|np.isnan(azm)
    notnan = np.logical_not(nan_ar)
    for i in range(len(alt)):
        if notnan[i]:
            all_loc.append([alt[i], el[i], azm[i]])

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

    #get the sensor location (lat, long, alt)
    lat = sensor_data[7][1]
    lon = sensor_data[8][1]
    sensor_alt = sensor_data[9][1]
    sensorloc = np.array([lat,lon,sensor_alt], dtype='f')
    coordnames = 'Spherical'
    
    return (data,coordnames,np.array(dataloc, dtype='f'),sensorloc,np.asarray(uniq_times, dtype='f'))
    
    
#data, coordnames, dataloc, sensorloc, times = readMad_hdf5('/Users/anna/Research/Ionosphere/2008WorldDaysPDB/son081001g.001.hdf5', ['ti', 'dti', 'nel'])

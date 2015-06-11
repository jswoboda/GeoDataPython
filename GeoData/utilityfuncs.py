#!/usr/bin/env python
"""
Created on Thu Sep 11 15:29:27 2014

@author: John Swoboda
"""
from __future__ import division,absolute_import
import pdb
import numpy as np
import tables as tb
import posixpath
import scipy as sp
from pandas import DataFrame
from warnings import warn
try:
    from . import CoordTransforms as CT
except Exception:
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
    with tb.openFile(filename, mode = "r", title = 'Sondrestrom1') as files:
        all_data = files.getNode('/Data/Table Layout').read()
        sensor_data = files.getNode('/Metadata/Experiment Parameters').read()


    instrument = str(sensor_data[0][1]) #instrument type string, comes out as bytes so cast to str
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
        exit("Error: Radar type not supported by program in this version.")

    # get the data location (range, el1, azm)
    if radar == 1:
        irng = 'gdalt'
    elif radar in (2,3):
        irng = 'range'

    all_loc = DataFrame(columns=['range','az','el','ut1','ut2'])
    try:
        all_loc['range'] = all_data[irng]
    except ValueError: pass
    try:
        all_loc['az'] = all_data['azm']
    except ValueError: pass
    try:
        all_loc['el'] = all_data['elm']
    except ValueError: pass

    all_loc['ut1'] = all_data['ut1_unix']
    all_loc['ut2'] = all_data['ut2_unix']

    for p in paramstr:
        all_loc[p] = all_data[p]

#%% SELECT
    all_loc.dropna(axis=0,how='any',subset=['range','az','el'],inplace=True)
    #notnan = all_loc.index

    #create list of unique data location lists
    dataloc = all_loc.drop_duplicates(subset=['range','az','el'])
    uniq_times = np.unique(all_loc['ut1'])

    #initialize and fill data dictionary with parameter arrays
    data = {}
    for p in paramstr:
        if not p in all_data.dtype.names:
            warn('{} is not a valid parameter name.'.format(p))
            continue
        data[p] = all_loc['nel'].reshape((dataloc.shape[0],uniq_times.size))
#        data[p]= DataFrame(index=[dataloc['range'],dataloc['az'],dataloc['el']],
#                            columns=uniq_times)
#        for i,qq in all_loc.iterrows():
#            ci = qq[['range','az','el']].values
#            data[p].loc[ci[0],ci[1],ci[2]][qq['ut1'].astype(int)] = qq[p]


    #get the sensor location (lat, long, rng)
    lat = sensor_data[7][1]
    lon = sensor_data[8][1]
    sensor_alt = sensor_data[9][1]
    sensorloc = np.array([lat,lon,sensor_alt], dtype='f')
    coordnames = 'Spherical'

    return (data,coordnames,np.array(dataloc, dtype='f'),sensorloc,np.asarray(uniq_times, dtype='f'))

def read_h5_main(filename):
    ''' Read in the structured h5 file.'''
    with tb.openFile(filename) as h5file:
        output={}
        # Read in all of the info from the h5 file and put it in a dictionary.
        for group in h5file.walkGroups(posixpath.sep):
            output[group._v_pathname]={}
            for array in h5file.listNodes(group, classname = 'Array'):
                output[group._v_pathname][array.name]=array.read()

    #pdb.set_trace()
    # find the base paths which could be dictionaries or the base directory
#    outarr = [pathparts(ipath) for ipath in output.keys() if len(pathparts(ipath))>0]
    outlist = {}
    basekeys  = output[posixpath.sep].keys()
    # Determine assign the entries to each entry in the list of variables.
    # Have to do this in order because of the input being a list instead of a dictionary
#%%
    #dictionary
    for ipath in output:
        if ipath[1:] in VARNAMES:
            outlist[ipath[1:]] = output[ipath]
            continue
    # for non-dicitonary
    for k in basekeys:
        if k in VARNAMES:
            # Have to check for MATLAB type strings, for some reason python does not like to register them as strings
            curdata = output['/'][k]
            if isinstance(curdata,np.ndarray):
                if curdata.dtype.kind=='S':
                    curdata=str(curdata)
            outlist[k] = curdata

    return outlist

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
    out = read_h5_main(filename)
    optical = out['data']
    enu = out['dataloc']
    dataloc = CT.enu2cartisian(enu)
    coordnames = 'Cartesian'
    sensorloc = out['sensorloc']
    times = out['times']

    return (optical, coordnames, dataloc, sensorloc, times)

def readIono(iono):
    """ This function will bring in instances of the IonoContainer class into GeoData.
    this is using the set up from my own code"""
    pnames = iono.Param_Names
    Param_List = iono.Param_List
    pdb.set_trace()
    (nloc,nt) = Param_List.shape[:2]
    if type(pnames) == sp.ndarray:
        if pnames.ndim>1:
            ionkeys = pnames.flatten()
            Param_List = sp.reshape(Param_List,(nloc,nt,len(ionkeys)))
        else:
            ionkeys=pnames
    else:
        ionkeys=pnames
    paramdict = {ikeys:Param_List[:,:,ikeyn] for ikeyn, ikeys in enumerate(ionkeys)}
    Nis = {}
    Tis = {}
    # Add Ti
    for ikey in ionkeys:
        if 'Ti_' ==ikey[:3]:
            Tis[ikey[3:]] = paramdict[ikey]
        elif 'Ni_' ==ikey[:3]:
            Nis[ikey[3:]] = paramdict[ikey]
    Nisum = sp.zeros((nloc,nt),dtype=Param_List.dtype)
    Ti = sp.zeros_like(Nisum)
    for ikey in Tis.keys():
        Ti =Tis[ikey]*Nis[ikey] +Ti
        Nisum = Nis[ikey]+Nisum
    if len(Ti)!=0:
        paramdict['Ti'] = Ti/Nisum
    if iono.Coord_Vecs == ['r','theta','phi']:
        coordnames = 'Spherical'
        coords = iono.Sphere_Coords
    elif iono.Coord_Vecs == ['x','y','z']:
        coordnames = 'Cartesian'
        coords = iono.Cart_Coords

    return (paramdict,coordnames,coords,sp.array(iono.Sensor_loc),iono.Time_Vector)

#data, coordnames, dataloc, sensorloc, times = readMad_hdf5('/Users/anna/Research/Ionosphere/2008WorldDaysPDB/son081001g.001.hdf5', ['ti', 'dti', 'nel'])

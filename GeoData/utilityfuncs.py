#!/usr/bin/env python
"""
Created on Thu Sep 11 15:29:27 2014

@author: John Swoboda
"""
from __future__ import division,absolute_import
from six import string_types,integer_types
import logging
import numpy as np
import tables as tb
import h5py
import posixpath
import scipy as sp
from astropy.io import fits
from pandas import DataFrame
from datetime import datetime
from dateutil.parser import parse
from pytz import UTC
from os.path import expanduser
#
from . import CoordTransforms as CT

USEPANDAS = True #20x speedup vs CPython
VARNAMES = ['data','coordnames','dataloc','sensorloc','times']

def readMad_hdf5 (filename, paramstr): #timelims=None
    """@author: Michael Hirsch / Anna Stuhlmacher
    madgrigal h5 read in function for the python implementation of GeoData for Madrigal Sondrestrom data
    Input:
    filename path to hdf5 file
    list of parameters to look at written as strings
    Returns:
    dictionary with keys are the Madrigal parameter string, the value is an array
    rows are unique data locations (data_loc) = (rng, azm, el1)
    columns are unique times

    Here we use Pandas DataFrames internally to speed the reading process by 20+ times,
    while still passing out Numpy arrays
    """
    filename = expanduser(filename)
    #open hdf5 file
    with h5py.File(filename, "r", libver='latest') as f:
        all_data = f['/Data/Table Layout'].value
        sensor_data = f['/Metadata/Experiment Parameters'].value


    instrument = str(sensor_data[0][1]) #instrument type string, comes out as bytes so cast to str
    if "Sondrestrom" in instrument:
        radar = 1
        logging.info("Sondrestrom data")
    elif "Poker Flat" in instrument:
        radar = 2
        logging.info("PFISR data")
    elif "Resolute Bay" in instrument:
        radar = 3
        logging.info("RISR data")
    else:
        raise NotImplementedError("Radar type "+str(instrument) +" not supported.")

    # get the data location (range, el1, azm)
    if radar == 1:
        irng = 'gdalt'
    elif radar in (2,3):
        irng = 'range'

    filt_data = DataFrame(columns=['range','az','el','ut1','ut2'])
    try:
        filt_data['range'] = all_data[irng]
    except ValueError: pass
    try:
        filt_data['az'] = all_data['azm']
    except ValueError: pass
    try:
        filt_data['el'] = all_data['elm']
    except ValueError: pass

    filt_data['ut1'] = all_data['ut1_unix']
    filt_data['ut2'] = all_data['ut2_unix']

    for p in paramstr:
        filt_data[p] = all_data[p]

#%% SELECT
    filt_data.dropna(axis=0,how='any',subset=['range','az','el'],inplace=True)

    #create list of unique data location lists
    dataloc = filt_data[['range','az','el']].drop_duplicates()
    uniq_times = filt_data['ut1'].drop_duplicates().values


    #initialize and fill data dictionary with parameter arrays
    #notnan = filt_data.index
    if not USEPANDAS:
        all_loc=filt_data[['range','az','el']].values.tolist()
        all_times = filt_data['ut1'].values.tolist()
        dataloclist = dataloc.values.tolist()
        uniq_timeslist = uniq_times = filt_data['ut1'].drop_duplicates().values.tolist()
        maxcols = len(uniq_times);  maxrows = len(dataloc)

    data = {}
    for p in paramstr:
        if not p in all_data.dtype.names:
            logging.warning('{} is not a valid parameter name.'.format(p))
            continue

        if USEPANDAS:
        # example of doing via numpy
        # filt_data has already been filtered for time and location with the isr parameter(s) riding along.
         #Just reshape it!
            #NOTE: take off the .values to pass the DataFrame
            data[p] = DataFrame(data=filt_data[p].reshape((dataloc.shape[0],uniq_times.shape[0]),order='F'),
                                               columns=uniq_times).values
        else:
            #example with CPython
            vec = filt_data[p].values #list of parameter pulled from all_data
            arr = np.empty([maxrows,maxcols]) #converting the tempdata list into array form

            for t in range(vec.size):
                #row
                row = dataloclist.index(all_loc[t])
                #column-time
                col = uniq_timeslist.index(all_times[t])
                arr[row][col] = vec[t]
            data[p] = arr

        #example of doing by MultiIndex
#        data[p]= DataFrame(index=[dataloc['range'],dataloc['az'],dataloc['el']],
#                            columns=uniq_times)
#        for i,qq in filt_data.iterrows():
#            ci = qq[['range','az','el']].values
#            data[p].loc[ci[0],ci[1],ci[2]][qq['ut1'].astype(int)] = qq[p]


    #get the sensor location (lat, long, rng)
    lat,lon,sensor_alt = sensor_data[7][1],sensor_data[8][1],sensor_data[9][1]
    sensorloc = np.array([lat,lon,sensor_alt], dtype=float) #they are bytes so we NEED float!
    coordnames = 'Spherical'
    #NOTE temporarily passing dataloc as Numpy array till rest of program is updated to Pandas
    return (data,coordnames,dataloc.values,sensorloc,uniq_times)

def readSRI_h5(filename,paramstr,timelims = None):
    '''This will read the SRI formated h5 files for RISR and PFISR.'''
    coordnames = 'Spherical'

        # Set up the dictionary to find the data
    pathdict = {'Ne':('/FittedParams/Ne', None),
                'dNe':('/FittedParams/Ne',None),
                'Vi':('/FittedParams/Fits',   (0,3)),
                'dVi':('/FittedParams/Errors',(0,3)),
                'Ti':('/FittedParams/Fits',   (0,1)),
                'dTi':('/FittedParams/Errors',(0,1)),
                'Te':('/FittedParams/Fits',  (-1,1)),
                'Ti':('/FittedParams/Errors',(-1,1))}

    with h5py.File(filename,'r',libver='latest') as f:
        # Get the times and time lims
        times = f['/Time/UnixTime'].value
        # get the sensor location
        sensorloc = np.array([f['/Site/Latitude'],
                              f['/Site/Longitude'],
                              f['/Site/Altitude']])
        # Get the locations of the data points
        rng = f['/FittedParams/Range'].value / 1e3
        angles = f['/BeamCodes'][:,1:2].value

    nt = times.shape[0]
    if timelims is not None:
        timelog = times[:,0]>= timelims[0] and times[:,1]<timelims[1]
        times = times[timelog,:]
        nt = times.shape[0]
#
    nrng = rng.shape[1]
    repangles = np.tile(angles,(1,2.0*nrng))
    allaz = repangles[:,::2]
    allel = repangles[:,1::2]
#   NOTE dataloc = DataFrame(index=times,
#                              {'rng':rng.ravel(),
#                               'allaz':allaz.ravel(),'allel':allel.ravel()})
    dataloc =np.vstack((rng.ravel(),allaz.ravel(),allel.ravel())).transpose()
    # Read in the data
    data = {}
    with h5py.File(filename,'r',libver='latest') as f:
        for istr in paramstr:
            if not istr in list(pathdict.keys()):
                logging.warning(istr + ' is not a valid parameter name.')

                continue
            curpath = pathdict[istr][0]
            curint = pathdict[istr][-1]

            if curint is None:
                tempdata = f[curpath].value
            else:
                tempdata = f[curpath][:,:,:,curint[0],curint[1]].value
            data[istr] = np.array([tempdata[iT,:,:].ravel() for iT in range(nt)]).transpose()

    return (data,coordnames,dataloc,sensorloc,times)

def read_h5_main(filename):
    '''
    Read in the structured h5 file.
    use caution with this function -- indexing dicts is less safe
    because the index order of dicts is not deterministic.
    '''
    with tb.openFile(filename) as h5file:
        output={}
        # Read in all of the info from the h5 file and put it in a dictionary.
        for group in h5file.walkGroups(posixpath.sep):
            output[group._v_pathname]={}
            for array in h5file.listNodes(group, classname = 'Array'):
                output[group._v_pathname][array.name]=array.read()

    # find the base paOMTIdata.h5ths which could be dictionaries or the base directory
#    outarr = [pathparts(ipath) for ipath in output.keys() if len(pathparts(ipath))>0]
    outlist = {}
    basekeys  = output[posixpath.sep].keys()
    # Determine assign the entries to each entry in the list of variables.
    # Have to do this in order because of the input being a list instead of a dictionary

    #dictionary
    for ipath in output:
        if ipath[1:] in VARNAMES:
            outlist[ipath[1:]] = output[ipath]
            continue
    # for non-dictionary
    for k in basekeys:
        if k in VARNAMES:
            # Have to check for MATLAB type strings, for some reason python does not like to register them as strings
            curdata = output['/'][k]
            if isinstance(curdata,np.ndarray):
                if curdata.dtype.kind=='S':
                    curdata=str(curdata)
            outlist[k] = curdata
    newout = [outlist[x] for x in VARNAMES]
    return newout

def pathparts(path):
    '''This will return all of the parts of a posix path in a list. '''
    components = []
    while True:
        (path,tail) = posixpath.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)

def readOMTI(filename, paramstr):
    """
    The data paths are known a priori, so read directly ~10% faster than pytables
    """
    filename = expanduser(filename)
    with h5py.File(filename,'r') as f:
        optical = {'optical':f['data/optical'].value} #for legacy API compatibility
        dataloc = CT.enu2cartisian(f['dataloc'].value)
        coordnames = 'Cartesian'
        sensorloc = f['sensorloc'].value.squeeze()
        times = f['times'].value

    return optical, coordnames, dataloc, sensorloc, times

def readIono(iono):
    """ @author:John Swoboda
    This function will bring in instances of the IonoContainer class into GeoData.
    This is using the set up from the RadarDataSim codebase"""
    pnames = iono.Param_Names
    Param_List = iono.Param_List
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
    if 'Ti' not in ionkeys:
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

def readAllskyFITS(flist,azmap,elmap,heightkm,sensorloc):
    """ @author: Michael Hirsch, Greg Starr
    For example, this works with Poker Flat DASC all-sky, FITS data available from:
    https://amisr.asf.alaska.edu/PKR/DASC/RAW/

    This function will read a Fits file into the proper GeoData variables.
    inputs
    flist - A list of Fits files that will be read in.
    azmap - A file name of the az mapping.
    elmap - A file name of the elevation maping
    hightkm - The height the data will be projected on to in km
    sensorloc - A 3-element vector of latitude, longitude and altitude in wgs coordinates of
    the location of the sensor.
    """
    if isinstance(flist,string_types):
        flist=[flist]
    assert isinstance(flist,(list,tuple)) and len(flist)>0
    assert isinstance(azmap,string_types)
    assert isinstance(elmap,string_types)
    assert isinstance(heightkm,(integer_types,float))
    assert isinstance(sensorloc,(tuple,list,np.ndarray)) and len(sensorloc)==3
#%% priming read
    with fits.open(flist[0],mode='readonly') as h:
        img = h[0].data
    dataloc = np.empty((img.size,3))
    times =   np.empty((len(flist),2))
    img =     np.zeros((len(flist),img.shape[0],img.shape[1]),img.dtype)
    epoch = datetime(1970,1,1,0,0,0,tzinfo=UTC)
#%% loop over files to read images
    for i,f in enumerate(flist):
        with fits.open(f,mode='readonly') as h:
            expstart_dt = parse(h[0].header['OBSDATE'] + ' ' + h[0].header['OBSSTART'])
            expstart_unix = (expstart_dt - epoch).total_seconds()
            times[i,:] = [expstart_unix,expstart_unix + h[0].header['EXPTIME']]
            img[i,...] = h[0].data
#%% get az/el calibration data
    coordnames="spherical"
    dataloc[:,0] = heightkm
    with fits.open(azmap,mode='readonly') as h:
        dataloc[:,1] = h[0].data.ravel()
    with fits.open(elmap,mode='readonly') as h:
        dataloc[:,2] = h[0].data.ravel()
#%% pack into GeoData class
    data = {'image':img}

    return (data,coordnames,dataloc,sensorloc,times)

def readNeoCMOS(imgfn, azelfn, heightkm=110.,treq=None):
    """
    treq is pair or vector of UT1 unix epoch times to load--often file is so large we can't load all frames into RAM.
    assumes that /rawimg is a 3-D array
    """
    assert isinstance(imgfn,string_types)
    assert isinstance(azelfn,string_types)
    assert isinstance(heightkm,(integer_types,float))
    imgfn = expanduser(imgfn)
    azelfn = expanduser(azelfn)
#%% load data
    with h5py.File(azelfn,'r',libver='latest') as f:
        az = f['/az'].value
        el = f['/el'].value

    with h5py.File(imgfn,'r',libver='latest') as f:
        times = f['/ut1_unix'].value
        sensorloc = f['/sensorloc'].value

        npix = np.prod(f['/rawimg'].shape[1:]) #number of pixels in one image
        dataloc = np.empty((npix,3),dtype=float)

        if treq is not None:
            mask = (treq[0]<times) & (times<treq[-1])
        else:
            mask = np.ones(f['/rawimg'].shape[0]).astype(bool)

        if mask.sum()*npix*2 > 1e9: #loading more than 1GByte into RAM
            logging.warning('trying to load very large amount of image data, your program may crash')

        imgs = f['/rawimg'][mask,...]
#%% plate scale
        if f['/params']['transpose']:
            imgs = imgs.transpose(0,2,1)
            az   = az.T
            el   = el.T
        if f['/params']['rotccw']: #NOT isinstance integer_types!
            imgs = np.rot90(imgs.transpose(1,2,0),k=f['/params']['rotccw']).transpose(2,0,1)
            az   = np.rot90(az,k=f['/params']['rotccw'])
            el   = np.rot90(el,k=f['/params']['rotccw'])
        if f['/params']['fliplr']:
            imgs = np.fliplr(imgs)
            az   = np.fliplr(az)
            el   = np.fliplr(el)
        if f['/params']['flipud']:
            imgs = np.flipud(imgs.transpose(1,2,0)).transpose(2,0,1)
            az   = np.flipud(az)
            el   = np.flipud(el)

    optical = {'optical':imgs}

    coordnames = 'Spherical'
    dataloc[:,0] = heightkm
    dataloc[:,1] = az.ravel()
    dataloc[:,2] = el.ravel()

    return optical, coordnames, dataloc, sensorloc, times[mask]


def readAVI(fn,fwaem):
    """
    caution: this was for a one-off test. Needs a bit of touch-up to be generalized to all files.
    """
    import cv2
    vid = cv2.VideoCapture(fn)
    width = vid.get(3)
    height = vid.get(4)
    fps = vid.get(5)
    fcount = vid.get(7)
    #data
    data=np.zeros((width*height,fcount))
    while 1:
        op,frame = vid.read()
        if not op:
            break
        data[:,vid.get(1)]=frame.flatten()
    data={'image':data}

    #coordnames
    coordnames="spherical"

    #dataloc
    dataloc=np.zeros((width*height,3))
    mapping = sio.loadmat(fwaem)
    dataloc[:,2]=mapping['el'].flatten()
    dataloc[:,1]=mapping['az'].flatten()
    dataloc[:,0]=120/np.cos(90-mapping['el'].flatten())

    #sensorloc
    sensorloc=np.array([65,-148,0])

    #times
    times=np.zeros((fcount+1,2))
    begin = (datetime(2007,3,23,11,20,5)-datetime(1970,1,1,0,0,0)).total_seconds()
    end = begin+fcount/fps
    times[:,0]=np.arange(begin,end,1/fps)
    times[:,1]=np.arange(begin+(1/fps),end+(1/fps),1/fps)

    return data,coordnames,dataloc,sensorloc,times

    vid = cv2.VideoCapture(fn)
    width = vid.get(3)
    height = vid.get(4)
    fps = vid.get(5)
    fcount = vid.get(7)
    #data
    data=np.zeros((width*height,fcount))
    while 1:
        op,frame = vid.read()
        if not op:
            break
        data[:,vid.get(1)]=frame.flatten()
    data={'image':data}

    #coordnames
    coordnames="spherical"

    #dataloc
    dataloc=np.zeros((width*height,3))
    mapping = sp.io.loadmat(fwaem)
    dataloc[:,2]=mapping['el'].flatten()
    dataloc[:,1]=mapping['az'].flatten()
    dataloc[:,0]=120/np.cos(90-mapping['el'].flatten())

    #sensorloc
    sensorloc=np.array([65,-148,0])

    #times
    times=np.zeros((fcount+1,2))
    begin = (datetime(2007,3,23,11,20,5)-datetime(1970,1,1,0,0,0)).total_seconds()
    end = begin+fcount/fps
    times[:,0]=np.arange(begin,end,1/fps)
    times[:,1]=np.arange(begin+(1/fps),end+(1/fps),1/fps)

    return data,coordnames,dataloc,sensorloc,times


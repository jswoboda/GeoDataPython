#!/usr/bin/env python
"""
Note: "cartesian" column order is x,y,z in the Nx3 matrix

This module holds a number of functions that can be used to read data into
GeoData objects. All of the function s have the following outputs
(data,coordnames,dataloc,sensorloc,times)
Outputs
    data - A dictionary with keys that are the names of the data. The values
        are numpy arrays. If the data comes from satilites the arrays are one
        dimensional. If the data comes from sensors that are not moving the
        values are NlxNt numpy arrays.
    coordnames - The type of coordinate system.
    dataloc - A Nlx3 numpy array of the location of the measurement.
    sensorloc - The location of the sensor in WGS84 coordinates.
    times - A Ntx2 numpy array of times. The first element is start of the
        measurement the second element is the end of the measurement.

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
#
from . import CoordTransforms as CT
from . import Path

USEPANDAS = True #20x speedup vs CPython
VARNAMES = ['data','coordnames','dataloc','sensorloc','times']

EPOCH = datetime(1970,1,1,0,0,0,tzinfo=UTC)

def readMad_hdf5 (filename, paramstr): #timelims=None
    """@author: Michael Hirsch / Anna Stuhlmacher
    madrigal h5 read in function for the python implementation of GeoData for Madrigal Sondrestrom data
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
    h5fn = Path(filename).expanduser()
#%% read hdf5 file
    with h5py.File(str(h5fn), "r", libver='latest') as f:
        lat,lon,sensor_alt = (f['/Metadata/Experiment Parameters'][7][1],
                              f['/Metadata/Experiment Parameters'][8][1],
                              f['/Metadata/Experiment Parameters'][9][1])

        D = f['/Data/Table Layout']

        filt_data = DataFrame(columns=['range','az','el','ut1','ut2'])

        try:
            filt_data['range'] = D['gdalt']
        except ValueError:
            filt_data['range'] = D['range']

        filt_data['az'] = D['azm']
        filt_data['el'] = D['elm']

        filt_data['ut1'] = D['ut1_unix']
        filt_data['ut2'] = D['ut2_unix']

        for p in paramstr:
            filt_data[p] = D[p]

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
        if not p in filt_data:
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
            vec = filt_data[p].values #list of parameter pulled from all data
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
    sensorloc = np.array([lat,lon,sensor_alt], dtype=float) #they are bytes so we NEED float!
    coordnames = 'Spherical'
    #NOTE temporarily passing dataloc as Numpy array till rest of program is updated to Pandas
    return (data,coordnames,dataloc.values,sensorloc,uniq_times)

def readSRI_h5(fn,params,timelims = None):
    assert isinstance(params,(tuple,list))
    h5fn = Path(fn).expanduser()
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

    with h5py.File(str(h5fn),'r',libver='latest') as f:
        # Get the times and time lims
        times = f['/Time/UnixTime'].value
        # get the sensor location
        sensorloc = np.array([f['/Site/Latitude'].value,
                              f['/Site/Longitude'].value,
                              f['/Site/Altitude'].value])
        # Get the locations of the data points
        rng = f['/FittedParams/Range'].value / 1e3
        angles = f['/BeamCodes'][:,1:3]

    nt = times.shape[0]
    if timelims is not None:
        times = times[(times[:,0]>= timelims[0]) & (times[:,1]<timelims[1]) ,:]
        nt = times.shape[0]
# allaz, allel corresponds to rng.ravel()
    allaz = np.tile(angles[:,0],rng.shape[1])
    allel = np.tile(angles[:,1],rng.shape[1])

    dataloc =np.vstack((rng.ravel(),allaz,allel)).T
    # Read in the data
    data = {}
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        for istr in params:
            if not istr in pathdict.keys(): #list() NOT needed
                logging.error('{} is not a valid parameter name.'.format(istr))
                continue
            curpath = pathdict[istr][0]
            curint = pathdict[istr][-1]

            if curint is None: #3-D data
                tempdata = f[curpath]
            else: #5-D data -> 3-D data
                tempdata = f[curpath][:,:,:,curint[0],curint[1]]
            data[istr] = np.array([tempdata[iT,:,:].ravel() for iT in range(nt)]).T

    # remove nans from SRI file
    nanlog = sp.any(sp.isnan(dataloc),1)
    keeplog = sp.logical_not(nanlog)
    dataloc = dataloc[keeplog]
    for ikey in data.keys():
        data[ikey]= data[ikey][keeplog]
    return (data,coordnames,dataloc,sensorloc,times)

def read_h5_main(filename):
    '''
    Read in the structured h5 file.
    use caution with this function -- indexing dicts is less safe
    because the index order of dicts is not deterministic.
    '''
    h5fn = Path(filename).expanduser()
    with tb.openFile(str(h5fn)) as f:
        output={}
        # Read in all of the info from the h5 file and put it in a dictionary.
        for group in f.walkGroups(posixpath.sep):
            output[group._v_pathname]={}
            for array in f.listNodes(group, classname = 'Array'):
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
    h5fn = Path(filename).expanduser()
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        optical = {'optical':f['data/optical'].value} #for legacy API compatibility
        dataloc = CT.enu2cartisian(f['dataloc'].value)
        coordnames = 'Cartesian'
        sensorloc = f['sensorloc'].value.squeeze()
        times = f['times'].value

    return optical, coordnames, dataloc, sensorloc, times

def readIono(iono,coordtype=None):
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
    # Get line of sight velocity
    if not 'Vi' in paramdict.keys():
        paramdict['Vi'] = iono.getDoppler()
    if coordtype is None:
        if iono.Coord_Vecs == ['r','theta','phi']:
            coordnames = 'Spherical'
            coords = iono.Sphere_Coords
        elif iono.Coord_Vecs == ['x','y','z']:
            coordnames = 'Cartesian'
            coords = iono.Cart_Coords
    elif coordtype.lower()=='cartesian':
        coordnames = 'Cartesian'
        coords = iono.Cart_Coords
    elif coordtype.lower() == 'spherical':
        coordnames = 'Spherical'
        coords = iono.Sphere_Coords
    return (paramdict,coordnames,coords,np.array(iono.Sensor_loc),iono.Time_Vector)

#data, coordnames, dataloc, sensorloc, times = readMad_hdf5('/Users/anna/Research/Ionosphere/2008WorldDaysPDB/son081001g.001.hdf5', ['ti', 'dti', 'nel'])

def readAllskyFITS(flist,azelfn,heightkm,timelims=[-sp.infty,sp.infty]):
    """ :author: Michael Hirsch, Greg Starr
    For example, this works with Poker Flat DASC all-sky, FITS data available from:
    https://amisr.asf.alaska.edu/PKR/DASC/RAW/

    This function will read a FITS file into the proper GeoData variables.

    inputs:
    ------
    flist - A list of Fits files that will be read in.
    azelfn - A tuple of file names  for az,el map files
    heightkm - The height the data will be projected on to in km
    timelims - A list of time limits in POSIX, the first element is the lower
       limit, the second is the upper limit.
    """
    azelfn = [Path(f).expanduser() for f in azelfn]

    if isinstance(flist,string_types):
        flist=[flist]

    assert isinstance(flist,(list,tuple)) and len(flist)>0, 'I did not find any image files to read'

    if azelfn is not None:
        assert isinstance(heightkm,(integer_types,float)), 'specify one altitude'
        assert isinstance(azelfn,(tuple,list)) and len(azelfn)==2, 'You must specify BOTH of the az/el files'
#%% priming read
    with fits.open(str(flist[0]),mode='readonly') as h:
        img = h[0].data
        sensorloc = np.array([h[0].header['GLAT'], h[0].header['GLON'], 0.]) #TODO real sensor altitude in km

    if isinstance(timelims[0],datetime):
        timelims = [(t-EPOCH).total_seconds() for t in timelims]

#%% search through the times to see if anything is between the limits
    times =[]
    flist2 = []
    for f in flist:
        try: #KEEP THIS try
            with fits.open(str(f),mode='readonly') as h:
                expstart_dt = parse(h[0].header['OBSDATE'] + ' ' + h[0].header['OBSSTART']+'Z') #implied UTC
                expstart_unix = (expstart_dt - EPOCH).total_seconds()
            if (expstart_unix>=timelims[0]) & (expstart_unix<=timelims[1]):
                times.append([expstart_unix,expstart_unix + h[0].header['EXPTIME']])
                flist2.append(f)
        except OSError as e:
            logging.info('trouble reading time from {}   {}'.format(f,e)) # so many corrupted files, we opt for INFO instead of WARNING
    times = np.array(times)
#%% read in the data that is in between the time limits
    img = np.empty((img.size,len(flist2)),dtype=img.dtype)  #len(flist2) == len(times)
    iok = np.zeros(len(flist2)).astype(bool)
    for i,f in enumerate(flist2):
        try:
            with fits.open(str(f),mode='readonly') as h:
                img[:,i] = np.rot90(h[0].data,1).ravel()

            iok[i] = True

            if not(i % 200) and i>0:
                print('{}/{} FITS allsky read'.format(i+1,len(flist2)))
        except OSError as e:
             logging.error('trouble reading images from {}   {}'.format(f,e))
#%% keep only good times
    img = img[:,iok]
    times = times[iok,:]
#%%
    coordnames = "spherical"

    if azelfn:
        with fits.open(str(azelfn[0]),mode='readonly') as h:
            az = h[0].data
        with fits.open(str(azelfn[1]),mode='readonly') as h:
            el = h[0].data
        #%% Get rid of bad data
        grad_thresh = 15.
        (Fx,Fy) = np.gradient(az)
        bad_datalog = np.hypot(Fx,Fy)>grad_thresh
        zerodata = bad_datalog | ((az==0.) & (el==0.))
        keepdata = ~(zerodata.ravel())
        optical = {'image':img[keepdata]}
        elfl = el.ravel()[keepdata]

        sinel = sp.sin(np.radians(elfl))
        dataloc = np.empty((keepdata.sum(),3))
        dataloc[:,0] = sp.ones_like(sinel)*heightkm/sinel #ALITUDE

        dataloc[:,1] = az.ravel()[keepdata]  # AZIMUTH
        dataloc[:,2] = el.ravel()[keepdata] # ELEVATION
    else: # external program
        az=el=dataloc=None
        optical = {'image':img}

    return optical,coordnames,dataloc,sensorloc,times

def readNeoCMOS(imgfn, azelfn, heightkm=None,treq=None):
    """
    treq is pair or vector of UT1 unix epoch times to load--often file is so large we can't load all frames into RAM.
    assumes that /rawimg is a 3-D array Nframe x Ny x Nx
    """
    #assert isinstance(heightkm,(integer_types,float))

    imgfn =  Path(imgfn).expanduser()
    azelfn = Path(azelfn).expanduser()
#%% load data
    with h5py.File(str(azelfn),'r',libver='latest') as f:
        az = f['/az'].value
        el = f['/el'].value

    with h5py.File(str(imgfn),'r',libver='latest') as f:
        times = f['/ut1_unix'].value
        sensorloc = f['/sensorloc'].value
        if sensorloc.dtype.fields is not None: #recarray
            sensorloc = sensorloc.view((float, len(sensorloc.dtype.names))).squeeze()

        npix = np.prod(f['/rawimg'].shape[1:]) #number of pixels in one image
        dataloc = np.empty((npix,3))

        if treq is not None:
            # note float() casts datetime64 to unix epoch for 'ms'
            if isinstance(treq[0],np.datetime64):
                treq = treq.astype(float)
            elif isinstance(treq[0],datetime):
                treq = np.array([(t-EPOCH).total_seconds() for t in treq])

            mask = (treq[0] <= times) & (times <= treq[-1])
        else: #load all
            mask = np.ones(f['/rawimg'].shape[0]).astype(bool)

        if mask.sum()*npix*2 > 1e9: # RAM
            logging.warning('trying to load {:.1f} GB of image data, your program may crash'.format(mask.sum()*npix*2/1e9))

        assert mask.sum()>0,'no times in {} within specified times.'.format(imgfn)
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

    coordnames = 'spherical'
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

#%% Mahali
def readIonofiles(filename):
    """iono file format
           1) time (as float day of year 0.0 - 366.0)
           2) year
           3) rec. latitude
           4) rec. longitude
           5) line-of-sight tec (TECu)
           6) error in line-of-sight tec (TECu)
           7) vertical tec (TECu)
           8) azimuth to satellite
           9) elevation to satellite
           10) mapping function (line of sight / vertical)
           11) pierce point latitude (350 km)
           12) pierce point longitude (350 km)
           13) satellite number (1-32)
           14) site (4 char)
           15) recBias (TECu)
           16) recBiasErr(TECu)
               #%f %f %f %f %f %f %f %f %f %f %f %f %f %s %s %s'
    """

#    fd = open(filename,'r')

#    data = np.loadtxt(fd,
#               dtype={'names': ('ToY', 'year', 'rlat', 'rlong', 'TEC', 'nTEC','vTEC','az','el','mf','plat','plon','sat','site','rbias','nrbias'),
#               'formats': ('float', 'float','float','float','float','float','float','float','float','float','float','float','float','S4', 'float','float')})
#    fd.close()
    data = np.genfromtxt(filename).T #NOTE this takes a long time, new data uses HDF5
    #%% Get in GeoData format
    doy = data[0]
    year=data[1].astype(int)

    if (year==year[1]).all():
        unixyear =(datetime(year[0],1,1,0,0,0,tzinfo=UTC) - EPOCH).total_seconds()
        uttime = unixyear+24*3600*sp.column_stack((doy,doy+1./24./60.)) # Making the difference in time to be a minute
    else:
        (y_u,y_iv) = np.unique(year,return_inverse=True)
        unixyearu = sp.array([(datetime(iy,1,1,0,0,0,tzinfo=UTC) - EPOCH).total_seconds() for iy in y_u])
        unixyear = unixyearu[y_iv]
        uttime = unixyear+24*3600*sp.column_stack((doy,doy+1))

    reclat = data[2]
    reclong = data[3]
    TEC = data[4]

    nTEC = data[5]

    vTEC = data[6]
    az2sat = data[7]
    el2sat = data[8]
    #mapfunc = data[9]

    piercelat = data[10]
    piercelong = data[11]
    satnum= data[12]
   # site = data[13]
    recBias = data[14]
    nrecBias = data[15]

    data = {'TEC':TEC,'nTEC':nTEC,'vTEC':vTEC,'recBias':recBias,'nrecBias':nrecBias,'satnum':satnum,'az2sat':az2sat,'el2sat':el2sat,
            'rlat':reclat,'rlong':reclong}
    coordnames = 'WGS84'
    sensorloc = sp.nan*sp.ones(3)
    dataloc = sp.column_stack((piercelat,piercelong,350e3*sp.ones_like(piercelat)))
    return (data,coordnames,dataloc,sensorloc,uttime)

def readMahalih5(filename,des_site):
    """ This function will read the mahali GPS data into a GeoData data structure.
        The user only has to give a filename and name of the desired site.
        Input
            filename - A string that holds the file name.
            des_site - The site name. Should be listed in the h5 file in the
                table sites.
    """
    h5fn = Path(filename).expanduser()

    with h5py.File(str(h5fn), "r", libver='latest') as f:

        despnts = sp.where(f['data']['site']==des_site)[0]
        # TODO: hard coded for now
        doy =  doy= f['data']['time'][despnts]
        year = 2015*sp.ones_like(doy,dtype=int)

        TEC = f['data']['los_tec'][despnts]

        nTEC = f['data']['err_los_tec'][despnts]

        vTEC = f['data']['vtec'][despnts]
        az2sat = f['data']['az'][despnts]
        el2sat = f['data']['az'][despnts]

        piercelat = f['data']['pplat'][despnts]
        piercelong = f['data']['pplon'][despnts]
        satnum= f['data']['prn'][despnts]
        recBias = f['data']['rec_bias'][despnts]
        nrecBias = f['data']['err_rec_bias'][despnts]

    # Make the integration time on the order of 15 seconds.
    if (year==year[1]).all():
        unixyear =(datetime(year[0],1,1,0,0,0,tzinfo=UTC) - EPOCH).total_seconds()
        uttime = unixyear + sp.round_(24*3600*sp.column_stack((doy,doy+15./24./3600.))) # Making the difference in time to be a minute
    else:
        (y_u,y_iv) = np.unique(year,return_inverse=True)
        unixyearu = sp.array([(datetime(iy,1,1,0,0,0,tzinfo=UTC) - EPOCH).total_seconds() for iy in y_u])
        unixyear = unixyearu[y_iv]
        uttime = unixyear + 24*3600*sp.column_stack((doy,doy+15./24./3600.))


    data = {'TEC':TEC,'nTEC':nTEC,'vTEC':vTEC,'recBias':recBias,'nrecBias':nrecBias,'satnum':satnum,'az2sat':az2sat,'el2sat':el2sat}
    coordnames = 'WGS84'
    sensorloc = sp.nan*sp.ones(3)
    dataloc = sp.column_stack((piercelat,piercelong, 350e3*sp.ones_like(piercelat)))

    return (data,coordnames,dataloc,sensorloc,uttime)

#!/usr/bin/env python
"""
Created on Thu Jul 17 12:46:46 2014

@author: John Swoboda
"""

import os
import time
import posixpath
import numpy as np
import scipy as sp
import scipy.interpolate as spinterp
import tables
import sys
import pdb
import CoordTransforms as CT
VARNAMES = ['data','coordnames','dataloc','sensorloc','times']
class GeoData(object):
    '''This class will hold the information for geophysical data.
    Variables
    data - This is a dictionary with strings for keys only. The strings are
    the given names of the data.
    coordnames - A string that holds the type of coordinate system.
    dataloc - A numpy array that holds the locations of the samples
    sensorloc - A numpy array with the WGS coordinates of the sensor.
    times - A numpy array that is holding the times associated with the measurements.'''
    def __init__(self,readmethod,inputs):
        '''This will create an instance of the GeoData class by giving it a read method and the inputs in a tuple'''
        (self.data,self.coordnames,self.dataloc,self.sensorloc,self.times) = readmethod(*inputs)
        # Assert that the data types are correct
        assert type(self.data) is dict,"data needs to be a dictionary"
        assert type(self.coordnames) is str, "coordnames needs to be a string"
        assert type(self.dataloc) is np.ndarray,"dataloc needs to be a numpy array"
        assert is_numeric(self.dataloc), "dataloc needs to be a numeric array"
        assert type(self.sensorloc) is np.ndarray,"sensorloc needs to be a numpy array"
        assert is_numeric(self.sensorloc), "sensorloc needs to be a numeric array"
        assert type(self.times) is np.ndarray,"times needs to be a numpy array"
        assert is_numeric(self.times), "times needs to be a numeric array"



    def interpolate(self,new_coords,newcoordname,method='linear',fill_value=np.nan):
        """This method will take the data points in the dictionary data and spatially.
        interpolate the points given the new coordinates. The method of interpolation
        will be determined by the input parameter method.
        Input:
            new_coords - A Nlocx3 numpy array. This will hold the new coordinates that
            one wants to interpolate the data over.
            newcoordname - New Coordinate system that the data is being transformed into.
            method - A string. The method of interpolation curently only accepts 'linear',
            'nearest' and 'cubic'
            fill_value - The fill value for the interpolation.
        """
        curavalmethods = ['linear', 'nearest', 'cubic']
        interpmethods = ['linear', 'nearest', 'cubic']
        if method not in curavalmethods:
            raise ValueError('Must be one of the following methods: '+ str(curavalmethods))
        Nt = self.times.shape[0]
        NNlocs = new_coords.shape[0]


        curcoords = self.__changecoords__(newcoordname)
        # Loop through parameters and create temp variable
        for iparam in self.data.keys():
            New_param = np.zeros((NNlocs,Nt),dtype=self.data[iparam].dtype)
            for itime in np.arange(Nt):

                curparam =self.data[iparam][:,itime]
                if method in interpmethods:
                    intparam = spinterp.griddata(curcoords,curparam,new_coords,method,fill_value)
                    New_param[:,itime] = intparam
            self.data[iparam] = New_param

        self.dataloc = new_coords
        self.coordnames=newcoordname

    def __changecoords__(self,newcoordname):

        if self.coordnames=='Spherical' and newcoordname=='Cartesian':
            return CT.sphereical2Cartisian(self.dataloc)
        if self.coordnames== 'Cartesian'and newcoordname=='Spherical':
            return CT.cartisian2Sphereical(self.dataloc)

        raise ValueError('Wrong inputs for coordnate names was given.')

    @staticmethod
    def read_h5(filename):
        """ Static method for this"""
        return GeoData(read_h5_main,[filename])

    def __eq__(self,self2):
        '''This is the == operator. '''
        # Check the data dictionary
        datakeys = self.data.keys()
        if set(datakeys) !=set(self2.data.keys()):
            return False

        for ikey in datakeys:
            a = np.ma.array(self.data[ikey],mask=np.isnan(self.data[ikey]))
            b = np.ma.array(self2.data[ikey],mask=np.isnan(self2.data[ikey]))
            if not np.ma.allequal(a,b):
                return False
        # Look at the coordinate names
        if self.coordnames!=self2.coordnames:
            return False
        # Look at the data location
#        pdb.set_trace()
        a = np.ma.array(self.dataloc,mask=np.isnan(self.dataloc))
        blah = np.ma.array(self2.dataloc,mask=np.isnan(self2.dataloc))
        if not np.ma.allequal(a,blah):
            return False
        # Look at the sensor location
        a = np.ma.array(self.sensorloc,mask=np.isnan(self.sensorloc))
        blah = np.ma.array(self2.sensorloc,mask=np.isnan(self2.sensorloc))
        if not np.ma.allequal(a,blah):
            return False
        # Look at the times
        a = np.ma.array(self.times,mask=np.isnan(self.times))
        blah = np.ma.array(self2.times,mask=np.isnan(self2.times))
        if not np.ma.allequal(a,blah):
            return False

        return True


    def __ne__(self,self2):
        '''This is the != operator. '''
        return not self.__eq__(self2)

    def datanames(self):
        '''Returns the data names.'''
        return self.data.keys()

    def write_h5(self,filename):
        '''Writes out the structured h5 files for the class.
        inputs
        filename - The filename of the output.'''
        h5file = tables.openFile(filename, mode = "w", title = "GeoData Out")
        # get the names of all the variables set in the init function
        varnames = self.__dict__.keys()
        vardict = self.__dict__
        try:
            # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
            # Make group for each dictionary
            for cvar in varnames:
                #group = h5file.create_group(posixpath.sep, cvar,cvar +'dictionary')
                if type(vardict[cvar]) ==dict: # Check if dictionary
                    dictkeys = vardict[cvar].keys()
                    group2 = h5file.create_group('/',cvar,cvar+' dictionary')
                    for ikeys in dictkeys:
                        h5file.createArray(group2,ikeys,vardict[cvar][ikeys],'Static array')
                else:
                    h5file.createArray('/',cvar,vardict[cvar],'Static array')
            h5file.close()

        except: # catch *all* exceptions
            e = sys.exc_info()
            h5file.close()
           # pdb.set_trace()
            print e
            sys.exit()

def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)
# TODO might want to make this private method
# currently just give this to the init function and it will create a class instance.
def read_h5_main(filename):
    ''' Read in the structured h5 file.'''
    h5file=tables.openFile(filename)
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
        for npath,ipath in enumerate(outarr):
            if ivar==ipath[0]:
                outlist.append(output[output.keys()[npath]])
                dictout=True
                break
        if dictout:
            continue

        for ikeys in basekeys:
            if ikeys==ivar:
                outlist.append(output[posixpath.sep][ikeys])

    return tuple(outlist)

def readSRI_h5(filename,paramstr,timelims = None):
    '''This will read the SRI formated h5 files for RISR and PFISR.'''
    coordnames = 'Spherical'
    h5file=tables.openFile(filename)
    # Set up the dictionary to find the data
    pathdict = {'Ne':('/FittedParams/Ne',None),'dNe':('/FittedParams/Ne',None),
                'Vi':('/FittedParams/Fits',(0,3)),'dVi':('/FittedParams/Errors',(0,3)),
                'Ti':('/FittedParams/Fits',(0,1)),'dTi':('/FittedParams/Errors',(0,1)),
                'Te':('/FittedParams/Fits',(-1,1)),'Ti':('/FittedParams/Errors',(-1,1))}

    # Get the times and time lims
    times = h5file.getNode('/Time/UnixTime').read()
    nt = times.shape[0]
    if timelims is not None:
        timelog = times[:,0]>= timelims[0] and times[:,1]<timelims[1]
        times = times[timelog,:]
        nt = times.shape[0]
    # get the sensor location
    lat = h5file.getNode('/Site/Latitude').read()
    lon = h5file.getNode('/Site/Longitude').read()
    alt = h5file.getNode('/Site/Altitude').read()
    sensorloc = np.array([lat,lon,alt])
    # Get the locations of the data points
    rng = h5file.getNode('/FittedParams/Range').read()/1e3
    angles = h5file.getNode('/BeamCodes').read()[:,1:2]
    nrng = rng.shape[1]
    repangles = np.tile(angles,(1,2.0*nrng))
    allaz = repangles[:,::2]
    allel = repangles[:,1::2]
    dataloc =np.vstack((rng.flatten(),allaz.flatten(),allel.flatten())).transpose()
    # Read in the data
    data = {}
    for istr in paramstr:
        if not istr in pathdict.keys():
            print 'Warning: ' +istr + ' is not a valid parameter name.'
            continue
        curpath = pathdict[istr][0]
        curint = pathdict[istr][-1]

        if curint is None:

            tempdata = h5file.getNode(curpath).read()
        else:
            tempdata = h5file.getNode(curpath).read()[:,:,:,curint[0],curint[1]]
        data[istr] = np.array([tempdata[iT,:,:].flatten() for iT in range(nt)]).transpose()
    h5file.close()
    return (data,coordnames,dataloc,sensorloc,times)

def pathparts(path):
    ''' '''
    components = []
    while True:
        (path,tail) = posixpath.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)
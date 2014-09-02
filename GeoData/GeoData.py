#!/usr/bin/env python
"""
Created on Thu Jul 17 12:46:46 2014

@author: Bodangles
"""

import os
import time
import posixpath
import numpy as np
import scipy as sp
import tables

VARNAMES = ['data','datanames','coordnames','dataloc','sensorloc','times']
class GeoData(object):
    '''This class will hold the information for geophysical data.'''
    # Variables
    # data - This is a dictionary with strings for keys only. The strings are the names of the
    def __init__(self,readmethod,inputs):
        '''This will create an instance of the GeoData class by giving it a read method and the inputs in a tuple'''
        (self.data,self.datanames,self.coordnames,self.dataloc,self.sensorloc,self.times) = readmethod(inputs)
        
    def write_h5(self,filename):
        h5file = tables.openFile(filename, mode = "w", title = "GeoData Out")
        # get the names of all the variables set in the init function
        varnames = self.__dict__.keys()
        vardict = self.__dict__
        
        # XXX only allow 1 level of dictionaries, do not allow for dictionary of dictionaries.
        # Make group for each dictionary
        for cvar in varnames:
            group = h5file.create_group(posixpath.sep, cvar)
            if type(vardict[cvar]) ==dict: # Check if dictionary
                dictkeys = vardict[cvar].keys()
                for ikeys in dictkeys:
                    group2 = h5file.create_group(os.path.join(cvar,ikeys))
                    h5file.createArray(group2,ikeys,vardict[cvar][ikeys],'Static array')
            h5file.createArray(group,cvar,vardict[cvar],'Static array')
        h5file.close()          
                    
# TODO might want to make this private method   
# currently just give this to the init function and it will create a class instance.
def read_h5(filename):
    h5file=tables.openFile(filename)
    output={}
    for group in h5file.walkGroups(posixpath.sep):
        output[group._v_pathname]={}
        for array in h5file.listNodes(group, classname = 'Array'):
            output[group._v_pathname][array.name]=array.read()
    h5file.close()
    outarr = [pathparts(ipath) for ipath in output.keys()]
    outlist = []
    for ivar in VARNAMES:
        firsttime = True
        dictout = False
        for npath,ipath in enumerate(outarr):
            if ivar==ipath[0]:
                if len(ipath)==1:
                    outlist.append(output[output.keys()[npath]])
                elif len(ipath)>1 and firsttime:
                    firsttime = False
                    dictout=True
                    curdict = {ipath[1]:output[output.keys()[npath]]}
                else:
                    curdict[ipath[1]] = output[output.keys()[npath]]
        if dictout:
            outlist.append(dictout)
    
    return tuple(outlist)
def readSRI_h5(filenames,timelims,):
    '''This will read the SRI formated h5 files for RISR and PFISR'''
    
def pathparts(path):
    components = [] 
    while True:
        (path,tail) = posixpath.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)
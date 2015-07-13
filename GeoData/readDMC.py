from rawDMCreader import goRead
import h5py
import numpy as np
from datetime import datetime

def readDMC(fn,fwaem):
    vid,a,b=goRead(fn,(512,512),(1,1))
    
    #data
    data=np.zeros((vid.shape[1]*vid.shape[2],vid.shape[0]))
    for i in range(vid.shape[0]):
        data[:,i]=vid[i].flatten()
    data={'image':data}
    
    #coordnames
    coordnames="spherical"
    
    #dataloc
    dataloc=np.zeros((vid.shape[1]*vid.shape[2],3))
    mapping = h5py.File(fwaem)
    az = np.array(mapping['az'])
    el = np.array(mapping['el'])
    ra = np.array(mapping['ra'])
    dataloc[:,2]=el.flatten()
    dataloc[:,1]=az.flatten()
    dataloc[:,0]=ra.flatten()
    
    #sensorloc
    sensorloc=np.array([65,-148,0])
    
    #times
    times=np.zeros((vid.shape[0],2))
    times[:,0]=(datetime(2013,4,14,8,54,0)-datetime(1970,1,1,0,0,0)).total_seconds()
    times[:,1]=(datetime(2013,4,14,8,54,0)-datetime(1970,1,1,0,0,0)).total_seconds()
    
    return data,coordnames,dataloc,sensorloc,times
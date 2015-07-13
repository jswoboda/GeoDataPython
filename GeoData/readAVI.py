import cv2
import numpy as np
import scipy.io as sio
from datetime import datetime
import GeoData.GeoData as gdata

def readAVI(fn,fwaem):   
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
    begin = (datetime(2007,03,23,11,20,05)-datetime(1970,1,1,0,0,0)).total_seconds()
    end = begin+fcount/fps
    times[:,0]=np.arange(begin,end,1/fps)
    times[:,1]=np.arange(begin+(1/fps),end+(1/fps),1/fps)
    
    return data,coordnames,dataloc,sensorloc,times
    
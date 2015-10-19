#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IQTools by John Swoboda

@author: John Swoboda
"""

import numpy as np


def CenteredLagProduct(rawbeams,N =14):
    """
    CenteredLagProduct
    by John Swoboda
    This function will create centered lag products.  It is assumed that the numpy
    array will be a [NxM] array where N is the number of range gates.
    
    """
    # It will be assumed the data will be range vs pulses    
    (Nr,Np) = rawbeams.shape    
    
    # Make masks for each piece of data
    arex = np.arange(0,N/2.0,0.5);
    arback = np.array([-int(np.floor(k)) for k in arex]);
    arfor = np.array([int(np.ceil(k)) for k in arex]) ;
    
    # figure out how much range space will be kept
    sp = max(np.abs(arback))+1;
    ep = Nr- np.max(arfor);
    rng_ar_all = np.arange(sp,ep);
    acf_cent = np.zeros((ep-sp,N))*(1+1j);
    for irng in  np.arange(len(rng_ar_all)):
        rng_ar1 =int(rng_ar_all[irng]) + arback
        rng_ar2 = int(rng_ar_all[irng]) + arfor
        # get all of the acfs across pulses # sum along the pulses
        acf_tmp = np.conj(rawbeams[rng_ar1,:])*rawbeams[rng_ar2,:]
        acf_ave = acf_tmp.sum(1)
        acf_cent[irng,:] = acf_ave# might need to transpose this
    return acf_cent
    
def FormatIQ(RAWDATA,des_samps):
    """
    FormatIQ
    by John Swoboda
    This will take the raw IQ data matrix [PxQxNx2] and pull out the desired samples and put it 
    into a final numpy array for use in the centered Lag Product.  The final format will be an
    array that is [NxL] where N is the desired pulses that will be integrated in the lag 
    product
    """
    ## Get into IQ data
    if len(des_samps)==2:
        Ipart =RAWDATA[des_samps[0],des_samps[1],:,0]
        Qpart = RAWDATA[des_samps[0],des_samps[1],:,1]
        IQdata = Ipart +1j*Qpart
    elif len(des_samps)==3:
        Ipart =RAWDATA[des_samps[0],des_samps[1],des_samps[2],0]
        Qpart = RAWDATA[des_samps[0],des_samps[1],des_samps[2],1]
        IQdata = Ipart +1j*Qpart
    ## Deal with cases where pulses go over multiple records
    if IQdata.ndim==2:
        # transpose the data so its in range x pulses format
        return IQdata.transpose()
    elif IQdata.ndim==3:
        (Nrec,Npul,Nrg)=IQdata.shape
        out_data = np.array([IQdata[:,:,x].flatten() for x in np.arange(Nrg)])
        return out_data
    else:
        raise   NameError('RAWDATA did not give the right amount of dimensions')
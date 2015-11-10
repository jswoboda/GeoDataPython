#!/usr/bin/env python
"""
spcorrbeamdecimate.py

John Swoboda
Michael Hirsch

original example:
python spcorrbeamdecimateall ~/U/eng_research_irs/ISRdata/20121207/20121207.002 --ext .dt2.h5 -b 65228 65288 65225 65291 -o ~/Documents/Python/data_experiments/OutData2
"""
from __future__ import division,absolute_import
from pathlib2 import Path
import numpy as np
import h5py
from os import makedirs,remove
import matplotlib.pyplot as plt
from tempfile import gettempdir
#
from GeoData import ioclass
from GeoData.IQTools import CenteredLagProduct, FormatIQ
#%% temporary hard-set parameters
npats = 10 #TODO: adapt to file contents
nrec_orig = 30

def spcorrbeam(h5fn,h5ext,beamids,outdir):
    h5fn = Path(h5fn).expanduser()
#%% Set up original beam patterns
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        nLags = f['/S/Data/Acf/Lags'].size
        Nranges = f['/S/Data/Acf/Range'].shape[1]
        test_data = f['/Raw11/Raw/RadacHeader/BeamCode'][:2,:].ravel()
        ptlen = f['/S/Data/Beamcodes'].shape[1]
        txbaud = f['/S/Data']['TxBaud'].value
        ambfunc = f['/S/Data']['Ambiguity'].value
        pwidth = f['/S/Data']['Pulsewidth'].value
    rclen = 20*ptlen
    pattern1 = np.array(beamids)

    fullpat = np.array([pattern1[x%4] for x in range(ptlen)])
#%% Output
#    lags = np.arange(0,nLags)*20e-6
#%% set up the output directory structure
    outpaths = mkoutdir(outdir,npats)
#%% Open and read file
    # Determine the start point
    stpnt = 0
    while True:
        subset_dat =test_data[stpnt:stpnt+ptlen]
        test_arr = subset_dat==fullpat
        if test_arr.all():
            break
        stpnt+=1
#%% Pull out the beam patterns
    # In this script x is used as the pattern iterator and y is used as the record indicator
    f_patternsdict = {(x):test_data[stpnt+x*2*ptlen:stpnt+(x+1)*2*ptlen] for x in range(npats) }
    # on repeation of the beams
    patternsdict = {x:f_patternsdict[x][:(x+2)**2] for x in f_patternsdict.keys()}

#%% Go through all the beams
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        all_beams_mat = f['/Raw11/Raw/RadacHeader/BeamCode'].value #keep this
#    all_beams = all_beams_mat.ravel() #need raveled and original
#    pnts = all_beams.size

    # to get the patterns just take themodulo with the file size afterward
    ## TODO Work on this when you get back##########################
    des_recs = 30
    #TODO should this be //rclen or /rclen

#    maxrecs = {x:(pnts-((x+1)*2*ptlen+stpnt))/rclen +1 for x in range(npats)}


    # determine the pattern
    patternlocdic = {(x):[(np.arange(x*2*ptlen+stpnt+y*rclen,(x+1)*2*ptlen+stpnt+y*rclen)) for y in range(des_recs)] for x in range(10)}
    ##########################################################

#%% Read the data and do set up
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        all_data = f['/Raw11/Raw/Samples/Data'].value
        rng = f['/Raw11/Raw/Samples/Range'][0,:]

    Nsamples = all_data.shape[2]

    # create the output ranges
    rngs = np.zeros((nLags,Nranges))
    for ilag in range(nLags-1,-1,-1):
        i1 = range(0,(Nsamples-ilag))
        i2 = range(ilag,Nsamples)
        ist = (i1[0]+i2[0])//2
        ist0 = (nLags-ilag-1)//2

        irng = range(0+(nLags-ilag-1),Nranges+(nLags-ilag-1))
        trng = (rng[:(Nsamples-ilag)] + rng[ilag:])/2.0

        rngs[ilag,:] = trng[ist0:][:Nranges]

    Nranges = Nsamples-nLags


#%% Need to make sure cal and noise data is correct shape
    with h5py.File(str(h5fn),'r',libver='latest') as f:
        beamcodes_cal =   f['/S/Cal/Beamcodes'].value
        beamcodes_noise = f['/S/Noise/Beamcodes'].value
    # do the checks
    #all columns == the first column
    cal_check = np.alltrue(np.array([np.alltrue(beamcodes_cal[:,ia]==beamcodes_cal[0,ia]) for ia in range(beamcodes_cal.shape[1])]))
    noise_check = np.alltrue(np.array([np.alltrue(beamcodes_noise[:,ia]== beamcodes_noise[0,ia]) for ia in range(beamcodes_noise.shape[1])]))
    if not cal_check:
        raise Exception('FAILED: Beams for cal are not consistant')
    if not noise_check:
        raise Exception('FAILED: Beams for noise are not consistant')
#%% first loop goes through patterns
    for x in range(npats):
        # set up the outputfiles
        curoutpath =outpaths[x]
        bname = h5fn.name
        spl = bname.split('.')
        oname = curoutpath / (spl[0]+'.' + spl[1] + '.proc.' + spl[2])

       # set up receivers and beams
        nrecs = len(patternlocdic[x])
        nbeams = len(patternsdict[x])

        # Checks to make sure he arrays are set

        #set up location arrays
        curbeams = patternsdict[x]
        cal_beam_loc = np.zeros(curbeams.shape,dtype=int)
        noise_beam_loc = np.zeros(curbeams.shape,dtype=int)
        cal_beam_loc = np.array([np.where(beamcodes_cal[0,:]==ib)[0][0] for ib in curbeams])
        noise_beam_loc = np.array([np.where(beamcodes_noise[0,:]==ib)[0][0] for ib in curbeams])

#        cal_pint = fullfiledict['/S/Cal']['PulsesIntegrated']
#        caldata = fullfiledict['/S/Cal/Power']['Data']
#        noise_pint = fullfiledict['/S/Noise']['PulsesIntegrated']
#        noise_pwer = fullfiledict['/S/Noise/Power']['Data']
#        noise_data =fullfiledict['/S/Noise/Acf']['Data']
#        # do all the call params
#        fullfiledict['/S/Cal']['Beamcodes'] = beamcodes_cal[:,cal_beam_loc]
#        fullfiledict['/S/Cal']['PulsesIntegrated'] = cal_pint[:,cal_beam_loc]
#        fullfiledict['/S/Cal/Power']['Data'] = caldata[:,cal_beam_loc]
#        # do all the noise params
#        fullfiledict['/S/Noise']['Beamcodes'] = beamcodes_noise[:,noise_beam_loc]
#        fullfiledict['/S/Noise']['PulsesIntegrated'] = noise_pint[:,noise_beam_loc]
#        fullfiledict['/S/Noise/Power']['Data'] = noise_pwer[:,noise_beam_loc]
#        fullfiledict['/S/Noise/Acf']['Data'] = noise_data[:,noise_beam_loc]
        irec = 0
        # second loop goes though all of the records
        for y in patternlocdic[x]:
            # determine the samples
            [arecs,asamps] = np.unravel_index(y,all_beams_mat.shape);
            # get the IQ data for all of the pulses in a pattern
            # this should keep the ordering
            fullIQ = FormatIQ(all_data,(arecs,asamps))

            # Beam by beam goes through the IQ data
            beamnum = 0
            # make holding arrays for acfs
            acf_rec = np.zeros((nbeams,nLags,Nranges,2))
            beams_rec =np.zeros((nbeams))
            pulsesintegrated = np.zeros((nbeams))
            pwr  = np.zeros((nbeams,Nranges))
            # fill in temp arrays
            for ibeam in patternsdict[x]:
                cur_beam_loc = np.where(f_patternsdict[x]==ibeam)[0]
                temp_lags = CenteredLagProduct(fullIQ[:,cur_beam_loc],nLags)
                acf_rec[beamnum,:,:,0] = temp_lags.real.transpose()
                acf_rec[beamnum,:,:,1] = temp_lags.imag.transpose()
                beams_rec[beamnum] = ibeam
                pulsesintegrated[beamnum] = len(cur_beam_loc)
                pwr[beamnum,] = acf_rec[beamnum,0,:,0]
                beamnum+=1

            # pack the files with data from each record
#            ofile.openFile()
#            ofile.createDynamicArray(ofile.h5Paths['Data_Power'][0]+'/Data',pwr)
#            ofile.createDynamicArray(ofile.h5Paths['Data_Acf'][0]+'/Data',acf_rec)
#            ofile.createDynamicArray(ofile.h5Paths['Data'][0]+'/PulsesIntegrated', pulsesintegrated)
#            ofile.createDynamicArray(ofile.h5Paths['Data'][0]+'/Beamcodes',beams_rec)
#            # pack the stuff that only is needed once
#            if irec ==0:
#                ofile.createDynamicArray(ofile.h5Paths['Data_Acf'][0]+'/Range',rngs[0,:])
#                ofile.createStaticArray(ofile.h5Paths['Data_Acf'][0]+'/Lags', lags[np.newaxis])
#                ofile.createDynamicArray(ofile.h5Paths['Data_Power'][0]+'/Range',rngs[0,:][np.newaxis])
#                ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/TxBaud',txbaud)
#                ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/Ambiguity',ambfunc)
#                ofile.createStaticArray(ofile.h5Paths['Data'][0]+'/Pulsewidth',pwidth)
#                # go through original file and get everything
#                for g_key in fullfiledict:
#                    cur_group = fullfiledict[g_key]
#                    for n_key in cur_group:
#
#                        if (nrecs < nrec_orig) and (type(cur_group[n_key])==np.ndarray):
#                            #kluge
#                            try:
#                                if cur_group[n_key].shape[0]==nrec_orig:
#                                    cur_group[n_key] = cur_group[n_key][1:]
#                            except:
#                                pass
#                        ofile.createStaticArray(ofile.h5Paths[g_key][0]+'/'+n_key,cur_group[n_key])
#            # close the file
#            ofile.closeFile()
#            irec+=1

        print('Data for Pattern '+str(x)+' has Finished')


def mkoutdir(outdir,npats):
    outdir = Path(outdir).expanduser()
    outpaths = {x:outdir/'Pattern{:02}'.format(x) for x in range(npats)}
    for x in np.arange(npats):
        makedirs(str(outpaths[x]),exist_ok=True)

    return outpaths

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='work with ISR lag products')
    p.add_argument('rootdir',help='directory containing ISR lag products *.dt2.h5') #dt2.h5
    p.add_argument('--ext',help='file extension of data files [.dt2.h5]',default='.dt3.h5')
    p.add_argument('-b','--beamid',help='beamid(s) to use [64157]',nargs='+',default=[64157],type=int)
    p.add_argument('-o','--outdir',help='directory in which to write the output',default=gettempdir())
    p = p.parse_args()

    spcorrbeam(p.rootdir,p.ext,p.beamid,p.outdir)


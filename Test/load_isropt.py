from GeoData import GeoData
from GeoData import utilityfuncs

def revpower(x1,x2):
    return x2**x1

def load_risromti(risrName,omtiName=None):
    #creating GeoData objects of the 2 files, given a specific parameter
    if omtiName:
        omti = GeoData.GeoData(utilityfuncs.readOMTI,(omtiName, ['optical']))
    else:
        omti = None

    risr = GeoData.GeoData(utilityfuncs.readMad_hdf5,(risrName, ['nel']))
    #converting logarthmic electron density (nel) array into electron density (ne) array
    risr.changedata('nel','ne',revpower,[10.0])

    return risr,omti

def load_pfisr_neo(isrName,optName=None,azelfn=None,heightkm=None):
    if optName:
        neo = GeoData.GeoData(utilityfuncs.readNeoCMOS,(optName,azelfn,heightkm))
    else:
        neo = None

    pfisr = GeoData.GeoData(utilityfuncs.readMad_hdf5,(isrName,['nel']))

    pfisr.changedata('nel','ne',revpower,[10.])

    return pfisr,neo
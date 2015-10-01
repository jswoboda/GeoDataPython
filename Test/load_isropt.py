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

def load_pfisr_neo(pfisrName,neoName=None):
    if neoName:
        neo = GeoData.GeoData(utilityfuncs.readNeo,(pfisrName,['nel']))
    else:
        neo = None

    pfisr = GeoData.GeoData(utilityfuncs.readMad_hdf5,(pfisrName['nel']))

    pfisr.changedata('nel','ne',revpower,[10.])

    return pfisr,neo
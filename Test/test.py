#!/usr/bin/env python3
"""
self-test for GeoDataPython
"""
from numpy.testing import assert_allclose,run_module_suite
#
from load_isropt import load_risromti

def test_risr():
    isrfn = 'data/ran120219.004.hdf5'
    omtifn = 'data/OMTIdata.h5'
    risr,omti = load_risromti(isrfn,omtifn)
    assert_allclose(risr.data['ne'][[36,136],[46,146]],
                    [119949930314.93805,79983425500.702927])
    assert_allclose(omti.data['optical'][[32,41],[22,39]],
                    [ 603.03568232,  611.20040632])
#test_risr()
run_module_suite()
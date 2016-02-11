#!/usr/bin/env python3
"""
self-test for GeoDataPython
"""
from numpy.testing import assert_allclose,run_module_suite
#
from load_isropt import load_risromti

def test_omti():
    fn = 'data/ran120219.004.hdf5'
    risr = load_risromti(fn)




run_module_suite()
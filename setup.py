#!/usr/bin/env python
"""
setup.py
This is the setup file for the GeoData python package.
To install as a package in development type 
python setup.py develop
To uninstall type
python setup.py develop --uninstall

@author: John Swoboda
"""
import os, inspect
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'GeoData class and needed functions.',
    'author': 'John Swoboda',
    'url': '',
    'download_url': 'https://github.com/jswoboda/GeoDataPython.git',
    'author_email': 'swoboj@bu.edu',
    'version': '0.2',
    'install_requires': ['numpy', 'scipy', 'tables'],
    'packages': ['GeoData'],
    'scripts': [],
    'name': 'GeoData'
}

curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testpath = os.path.join(curpath,'Test')
if not os.path.exists(testpath):
    os.mkdir(testpath)
    print("Making a path for testing at "+testpath)
setup(**config)
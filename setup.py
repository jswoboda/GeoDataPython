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
from setuptools import setup


with open('README.rst') as f:
	long_description = f.read()

config = {
    'description': 'GeoData class and needed functions.',
    'author': 'John Swoboda',
    'url': '',
    'download_url': 'https://github.com/jswoboda/GeoDataPython.git',
    'author_email': 'swoboj@bu.edu',
    'version': '0.2',
    'install_requires': [],
    'extras_require': {'mayavi2':'mayavi2'},
    'packages': ['GeoData'],
    'scripts': [],
    'name': 'GeoData',
    'long_description': long_description
}

curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
testpath = os.path.join(curpath,'Test')
if not os.path.exists(testpath):
    os.mkdir(testpath)
    print("Making a path for testing at "+testpath)
setup(**config)

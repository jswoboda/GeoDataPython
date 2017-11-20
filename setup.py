#!/usr/bin/env python
"""
setup.py
This is the setup file for the GeoData python package.
To install as a package in development: pip install -e .
To uninstall: pip uninstall .

@author: John Swoboda
"""
req = ['nose','six','python-dateutil','pathlib2','tables', 'h5py','pandas', 'numpy','scipy']

from setuptools import setup,find_packages

setup(description='GeoData class and needed functions.',
      author='John Swoboda',
      version='0.2.0',
      install_requires=req,
      extras_require={'plot':['mayavi','matplotlib','seaborn'],
                      'io':['beautifulsoup4','astropy'],},
      packages=find_packages(),
      name='GeoData',
      python_requires='>=2.7',
)


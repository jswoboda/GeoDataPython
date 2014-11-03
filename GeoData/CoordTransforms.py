#!/usr/bin/env python
"""
Coordinate Transforms
This will hold a number of different functions that can be used for coordinate transforms.
Created on Sat Nov  1 19:09:47 2014

@author: John Swoboda
"""

import numpy as np

def sphereical2Cartisian(spherecoords):
    d2r = np.pi/180.0
    (dir1,dir2) = spherecoords.shape
    transcoords = False
    if dir2==3:
        spherecoords = np.transpose(spherecoords)
        transcoords  = True
    if 3 not in spherecoords.shape:
        raise ValueError('Neither of the dimensions are of length 3')
    (R,Az,El) = spherecoords[:]

    Azr = Az*d2r
    Elr = El*d2r

    kx = np.sin(Azr) * np.cos(Elr)
    ky = np.cos(Azr) * np.cos(Elr)
    kz = np.sin(Elr)

    x = R*kx
    y = R*ky
    z = R*kz

    cartcoords = np.array([x,y,z])

    if transcoords:
        cartcoords = np.transpose(cartcoords)
    return cartcoords

def cartisian2Sphereical(cartcoords):
    """This function will """
    r2d = 180.0/np.pi
    (dir1,dir2) = cartcoords.shape
    transcoords = False
    if dir2==3:
        cartcoords = np.transpose(cartcoords)
        transcoords  = True
    if 3 not in cartcoords.shape:
        raise ValueError('Neither of the dimensions are of length 3')
    (x,y,z) = cartcoords[:]

    R = np.sqrt(x**2+y**2+z**2)
    Az = np.arctan2(y,x)*r2d
    El = np.arcsin(z/R)*r2d

    spherecoords = np.array([R,Az,El])

    if transcoords:
        spherecoords=np.transpose(spherecoords)
    return spherecoords

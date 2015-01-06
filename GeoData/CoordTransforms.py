#!/usr/bin/env python
"""
Coordinate Transforms
This will hold a number of different functions that can be used for coordinate transforms.
Created on Sat Nov  1 19:09:47 2014

@author: John Swoboda
"""

import numpy as np

def sphereical2Cartisian(spherecoords):
    """This function will convert Spherical coordinates to Cartisian coordinates.
    Input
    spherecoords - A 3xN numpy array with rows of range (in km) azimuth (in degrees)
    and elevation (in degrees).
    Output
    cartcoords - A 3xN numpy array with X, Y and Z in a cartisian coordinate space.
    The coordinates are in units of kilometers."""
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
    """This function will convert Cartisian coordinates to Spherical coordinates.
    Input
    cartcoords - A 3xN numpy array with X, Y and Z in a Cartisian coordinate space.
    The coordinates are in units of kilometers.
    Output
    spherecoords - A 3xN numpy array with rows of range (in km) azimuth (in degrees)
    and elevation (in degrees)."""
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

def wgs2ecef(WGS_COORDS):
    """ wgs2ecef
        ECEF_COORDS = wgs2ecef(WGS_COORDS)
        by John Swoboda 1/6/2015
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Description
        This function takes a set of latitude longitude height coordinates and
        puts them into ECEF.  The input is assumed to be a 3xN (or Nx3) matrix of
        WGS coordinates.  It then outputs a 3xN matrix of ECEF coordinates in
        meters. This has been compared to a working matlab version that
        has been comapred to the internal MATLAB function geodetic2ecef and has
        comparible results with a normalized difference on the order of 10^-33.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Input
        WGS_COORDS - A 3xN numpy array (can also be a Nx3 array) with the latitude
        longitude and height coordinates. The matrix is broken up in the
        following way, the first row is latitude in degrees, the second row is
        longitude in degrees and the last row is height from the surface of the
        earth in meters.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Output
        ECEF_COORDS -A 3xN numpy array with X, Y and Z in ECEF coordinate space.  The
        coordinates are in units of meters.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Reference
        Vermeille H (2002) Direct transformation from geocentric coordinates
        to geodetic coordinates, Journal of Geodesy, vol. 76, no. 8, pp 451 - 454,
        Nov 2002, http://link.springer.com/article/10.1007%2Fs00190-002-0273-6
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    #%% Check Input
    d2r = np.pi/180.0
    (dir1,dir2) = WGS_COORDS.shape
    transcoords = False
    if dir2==3:
        WGS_COORDS = np.transpose(WGS_COORDS)
        transcoords  = True
    if 3 not in WGS_COORDS.shape:
        raise ValueError('Neither of the dimensions are of length 3')

    #%% Get Lat, Long and Height
    phi = WGS_COORDS[0,:]*d2r
    lamb = WGS_COORDS[1,:]*d2r
    h = WGS_COORDS[2,:]

    #%% Set the constants
    a = 6378137.0 # semi-major axis in meters
    f = 1/298.257223563 # the flattening factor
    b = a*(1-f)# semiminor axis in meters

    e = np.sqrt(a**2-b**2)/a;# first eccentricity

    M_e = (a*(1-e**2))/(1-e**2*np.sin(phi))**(3.0/2.0)# Meridian radius of curvature
    n =  a/np.sqrt(1-e**2*np.sin(phi)**2) # prime verticl radius of curvature
    #%% Final Transform
    x_ecef = (n + h)*np.cos(phi)*np.cos(lamb);
    y_ecef = (n + h)*np.cos(phi)*np.sin(lamb);
    z_ecef = (n*(1-e**2)+h)*np.sin(phi);

    ECEF_COORDS = np.vstack((x_ecef,y_ecef,z_ecef));
    if transcoords:
        ECEF_COORDS = np.transpose(ECEF_COORDS);
    return ECEF_COORDS
def ecef2wgs(ECEF_COORDS):
    """ ecef2wgs
        ECEF_COORDS = wgs2ecef(WGS_COORDS)
        by John Swoboda 1/6/2015
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Description
        This function takes a set of X,Y Z ECEF coordinates and puts them into
        WGS84 coordinate of latitude longitude and height from the surface of the
        earth. The input is assumed to be a 3xN (or Nx3) matrix of ECEF
        coordinates in meters. It then outputs a 3xN matrix of WGS coordinates
        in degrees and meters. This has been compared to a working matlab version that
        has been comapred to the internal MATLAB function ecef2geodetic and has
        comparible results with a normalized difference on the order of 10^-23.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Input
        ECEF_COORDS - A 3xN numpy array (can also be a Nx3 array) with X, Y and Z in
        the ECEF coordinate space.  Coordinates are in units of meters.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Output
        WGS_COORDS - A 3xN numpy array  with the latitude longitude and height
        coordinates.  The matrix is broken up in the following way, the first row
        is latitude in degrees, the second row is longitude in degrees and the
        last row is height from the surface of the earth in meters.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Reference
        Vermeille H (2002) Direct transformation from geocentric coordinates
        to geodetic coordinates, Journal of Geodesy, vol. 76, no. 8, pp 451 - 454,
        Nov 2002, http://link.springer.com/article/10.1007%2Fs00190-002-0273-6
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
    #%% Check Input
    r2d = 180.0/np.pi
    (dir1,dir2) = ECEF_COORDS.shape
    transcoords = False
    if dir2==3:
        ECEF_COORDS = np.transpose(ECEF_COORDS)
        transcoords  = True
    if 3 not in ECEF_COORDS.shape:
        raise ValueError('Neither of the dimensions are of length 3')
    #%% Get Lat, Long and Height
    X = ECEF_COORDS[0,:]
    Y = ECEF_COORDS[1,:]
    Z = ECEF_COORDS[2,:]
    #%% Set the constants
    a = 6378137.0 # semi-major axis in meters
    f = 1/298.257223563 # the flattening factor
    b = a*(1-f)# semiminor axis in meters
    e = np.sqrt(a**2-b**2)/a# first eccentricity

    #%% Algorithm
    # this is taken straight from the Vermeille paper
    p = (X**2.0+Y**2.0)/a**2.0

    q = ((1-e**2.0)/a**2.0)*Z**2.0;

    r = (p+q-e**4.0)/6;

    s = e**4.0*(p*q)/(4.0*r**3.0);

    t = nthroot(1+s+np.sqrt(s*(2.0+s)),3);

    u = r*(1+t+(1/t))

    v = np.sqrt(u**2.0+(e**4)*q);

    w = e**2.0*((u+v-q)/(2.0*v));

    k = np.sqrt(u+v+w**2.0)-w;

    D = k*np.sqrt(X**2.0+Y**2.0)/(k+e**2.0);

    #%% Final Form
    # use the atan2 function for more numerical stability
    long = np.arctan2(Y,X)*r2d

    lat = np.arctan2(Z,D)*r2d

    h = ((k+e**2.0-1)/(k))*np.sqrt(D**2.0+Z**2.0)
    # put into the final form
    WGS_COORDS = np.vstack((lat,long,h));
    if transcoords:
        WGS_COORDS = np.transpose(WGS_COORDS)
    return WGS_COORDS

def enu2cartisian(ENUCOORDS):
    """ This function will transform enu coordinates to Cartisian coordinates, the only
    difference being that ENU coordinates are in meters and Cartisian are in km.
    Input
    ENUCOORDS - A 3xN numpy array with X, Y and Z in a ENU coordinate space.
    The coordinates are in units of meters.
    Output
    CARTCOORDS - A 3xN numpy array with X, Y and Z in a Cartisian coordinate space.
    The coordinates are in units of kilometers."""
    return ENUCOORDS*1e-3

def cartisian2enu(CARTCOORDS):
    """ This function will transform Cartisian coordinates to ENU coordinates, the only
    difference being that ENU coordinates are in meters and Cartisian are in km.
    Input
    CARTCOORDS - A 3xN numpy array with X, Y and Z in a Cartisian coordinate space.
    The coordinates are in units of kilometers.
    Output
    ENUCOORDS - A 3xN numpy array with X, Y and Z in a ENU coordinate space.
    The coordinates are in units of meters."""
    return CARTCOORDS*1e3

def nthroot(X,N):

    xlog = X<0
    xroot = np.abs(X)**(1.0/N)
    xroot[xlog] = -xroot[xlog]
    return xroot

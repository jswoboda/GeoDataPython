#!/usr/bin/env python
"""
This module will be used for accessing files from different websites
@author: John Swoboda
"""
from __future__ import division,absolute_import
from six.moves.urllib.request import urlretrieve,urlopen
from bs4 import BeautifulSoup
import re
import datetime
from dateutil import parser
import os
from pytz import UTC

def datedwebsite(baseurl,daterange,basedir=''):
    """This function will download a set of files from a directory structure based
    on the timestamp in the file name. Currently the directory structure must be the
    following Year/date/file.fits.
    Inputs
    baseurl - The URL for the base webpage must have folders with years in it.
    daterange - A list of dates that you want to get data from. Must be in posix format.
    basedir - The directory the files will be places after they're downloaded."""

    # Get date info from the ranges
    datet1 = datetime.datetime.fromtimestamp(daterange[0],tz=UTC)
    datet2 = datetime.datetime.fromtimestamp(daterange[1],tz=UTC)
    yearset = {datet1.year,datet2.year}
    dateset = {datet1.date(),datet2.date()}

    # Get the base directory information
    html = urlopen(baseurl)
    bsObj = BeautifulSoup(html)
    yearlist = []
    for link in bsObj.find_all("a",href=re.compile("^(20)")):
        if 'href' in link.attrs:
            yearlist.append(link.attrs['href'])
    yearnum = [int(iyr.strip('/')) for iyr in yearlist]

    # traverse directory for each year
    for iyr in yearset:
        if iyr not in yearnum:
            continue
        curindx = yearnum.index(iyr)
        yearurl = baseurl+yearlist[curindx]

        htmlyr = urlopen(yearurl)
        bsObjyr = BeautifulSoup(htmlyr)
        datelist = []
        for link in bsObjyr.find_all("a",href=re.compile("^(20)")):
            if 'href' in link.attrs:
                datelist.append(link.attrs['href'])
        datelistdt = [parser.parse(idate.strip('/')).date() for idate in datelist]

    #traverse directory for each date
        filelist = []
        fileurls = []
        for idate in dateset:
            if idate not in datelistdt:
                continue
            curindxdt = datelistdt.index(idate)
            dateurl = yearurl+datelist[curindxdt]

            htmldt = urlopen(dateurl)
            bsObjdt = BeautifulSoup(htmldt)
            for link in bsObjdt.find_all("a",href=re.compile("^(PKR)")):
                if 'href' in link.attrs:

                    pkfile = link.attrs['href']

                    pkfilesp = pkfile[:-5].split('_')
                    filetime = (parser.parse(pkfilesp[-2]+' '+pkfilesp[-1]))

                    # Determine if a file will be downloaded
                    if filetime>=datet1 and filetime<datet2:
                        fileurls.append(dateurl+pkfile)
                        filelist.append(pkfile)

    # Create directory and download the files
    try:
        os.mkdir(basedir)
    except:
        pass

    print('Downloading {:d} files to {}'.format(len(fileurls),basedir))
    for ifile in zip(filelist,fileurls):
        urlretrieve(ifile[1],os.path.join(basedir,ifile[0]))

def genwebscraperallsky(baseurl,daterange,basedir=''):
    """ """

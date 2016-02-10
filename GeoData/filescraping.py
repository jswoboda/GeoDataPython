#!/usr/bin/env python
"""
This module will be used for accessing files from different websites
@author: John Swoboda
"""
try:
    from urllib import urlretrieve, urlopen
except:
    from urllib.request import urlretrieve, urlopen
from bs4 import BeautifulSoup
import re
import datetime
from dateutil import parser
import os
import requests
requests.packages.urllib3.disable_warnings()

def datedwebsite(baseurl,daterange,basedir=''):
    """This function will download a set of files from a directory structure based
    on the timestamp in the file name. Currently the directory structure must be the
    following Year/date/file.fits.
    Inputs
    baseurl - The URL for the base webpage must have folders with years in it.
    daterange - A list of dates that you want to get data from. Must be in posix format.
    basedir - The directory the files will be places after they're downloaded."""

    # Get date info from the ranges
    datet1 = datetime.datetime.fromtimestamp(daterange[0])
    datet2 = datetime.datetime.fromtimestamp(daterange[1])
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
    if not os.path.exists(basedir) and basedir!='':
        os.mkdir(basedir)

    print 'Downloading {0:d} files to {1}'.format(len(fileurls),basedir)
    for ifile in zip(filelist,fileurls):
        urlretrieve(ifile[1],os.path.join(basedir,ifile[0]))

def genwebscraperallsky(baseurl,daterange,basedir=''):
    """ """
    
    
def download(urls,dldir):
   if not os.path.exists(dldir) and dldir!='':
        os.mkdir(dldir)
        
   for url in urls:
       f = open(os.path.join(dldir,url.split('/')[-1]),'wb')
       for i in requests.get(url,verify=False).iter_content():
           f.write(i)
       f.close()
       print urls.index(url),'out of',len(urls)
    
    
def searchsite(date1,date2,url):

    dlurls=[]
    
    date1 = datetime.datetime.fromtimestamp(date1)
    date2 = datetime.datetime.fromtimestamp(date2)
    
    soup = BeautifulSoup(requests.get(url,verify=False).content)
    yearlist = []
    for link in soup.find_all("a",href=re.compile("^(20)")):
        if 'href' in link.attrs:
            yearlist.append(link.attrs['href'])
    yearnum = [int(iyr.strip('/')) for iyr in yearlist]
    
    yearurls=[]
    for i in range(len(yearnum)):
        if yearnum[i]>=date1.year and yearnum[i]<=date2.year:
            yearurls.append(yearlist[i])
    print str(len(yearurls))+' years'  
    for year in yearurls:
        dates=[]
        dateurls=[]
        ydir = BeautifulSoup(requests.get(url+year,verify=False).content)
        for link in ydir.find_all("a",href=re.compile("^(20)")):
            if 'href' in link.attrs:
                dates.append(link.attrs['href'])
                
        for i in range(len(dates)):
            d1 = datetime.date(int(dates[i][:4]),int(dates[i][4:6]),int(dates[i][6:8]))
            if d1 >= date1.date() and d1 <= date2.date():
                dateurls.append(dates[i])
                
        print str(len(dateurls))+' dates from '+year
        for date in dateurls:
            fdir=BeautifulSoup(requests.get(url+year+date,verify=False).content)
            for link in fdir.find_all("a",href=re.compile("^(PKR)")):
                if 'href' in link.attrs:
                    pkfile = link.attrs['href']
                    pkfilesp = pkfile[:-5].split('_')
                    filetime = (parser.parse(pkfilesp[-2]+' '+pkfilesp[-1]))
                    if filetime>=date1 and filetime<=date2:
                        dlurls.append(url+year+date+pkfile)
                        
    return dlurls

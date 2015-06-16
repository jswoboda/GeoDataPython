##GeoDataPython
![alt text](https://raw.github.com/jswoboda/GeoDataPython/master/logo/logo1.png "GeoDataPython")

#Overview
This is the repository for the Python version of GeoData to plot and analyze data from geophysics sources such as radar and optical systems.

# Requirements
This runs on Python 2.7 / 3.4, except 3-D plotting which requires 2.7 due to Mayavi. The packages required include

* numpy
* scipy
* pytables
* matplotlib
* h5py
* pandas
* VTK
* Mayavi

#Installation
Installation is optional; in any case, clone repository:

	$ git clone https://github.com/jswoboda/GeoDataPython.git

If you which to install,

	$ cd GeoDataPython
	$ python setup.py develop

If you wish to use WITHOUT installing, install prereqs with either of:

	$ conda install --file requirements.txt

or 

	$ pip install -r requirements.txt

#Software Structure

The code is broken up these files: GeoData.py, utilityfuncs.py plotting.py and CoordTransforms.py. GeoData.py holds code related to the GeoData class such as the class def file and any other functions that directly use or impact the class. The file utilityfuncs.py hold code that will be used to read in data from new data types. The file plotting.py holds functions to plot data. Lastly the file CoordTransforms.py holds code for different coordinate transforms.

###Style Guide:

his style guide will cover conventions and elements specific to this codebase. For more general tips on proper Python style guidelines see PEP 8 the de-facto Python style guide (http://legacy.python.org/dev/peps/pep-0008/).

Code that impacts the class will be placed in the class def file. For example if it is desired to interpolate the data to a different set of coordinates this should be implemented as a method in the class. The code for doing this can be written outside of the class def file if needed but this should only be done to keep the code neat.

The read functions will be placed in utilityfuncs.py file. The names of the functions will start with read_ and then followed by a descriptive name.


The properties names will be all lower case. While all function names will be lower case with _ to separate words. The classes will have capitalized words with no spaces. Directories will be capitalized with _ to separate words.

If the user would like to create test code please do this in the Test folder. Also this code is not be uploaded to the main code base on GitHub. 

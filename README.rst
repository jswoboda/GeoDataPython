=============
GeoDataPython
=============

.. image:: logo/logo1.png
   :alt: GeoDataPython example of OMTI and RISR data



Overview
========
This is the repository for the Python version of GeoData to plot and analyze data from geophysics sources such as radar and optical systems.

Requirements
============
Full functionality requires Python 2.7 due to Mayavi/VTK.
Non-mayavi functionality also available in Python 3.
The packages required include:

* numpy
* scipy
* pytables
* matplotlib
* Mayavi (inlcludes VTK)

Installation
============
To install first clone repository:

	$ git clone https://github.com/jswoboda/GeoDataPython.git

Then move to the main directory and run the Python setup script, which should be run in develop mode

	$ cd GeoDataPython

	$ python setup.py develop

Software Structure
==================

The code is broken up into these files:

==================  ==============
File        				Functionality
==================  ==============
GeoData.py  				code related to the GeoData class such as the class def file and any other functions that directly use or impact the class.
utilityfuncs.py 		code that will be used to read in data from new data types.
plotting.py 				functions to plot data.
CoordTransforms.py  code for different coordinate transforms.
==================  ==============

Style Guide:
------------

This style guide will cover conventions and elements specific to this codebase. For more general tips on proper Python style guidelines see PEP 8 the de-facto `Python style guide <http://legacy.python.org/dev/peps/pep-0008/>`_.

Code that impacts the class will be placed in the class def file. For example if it is desired to interpolate the data to a different set of coordinates this should be implemented as a method in the class. The code for doing this can be written outside of the class def file if needed but this should only be done to keep the code neat.

The read functions will be placed in ``utilityfuncs.py`` file. The names of the functions will start with read_ and then followed by a descriptive name.


The properties names will be all lower case. While all function names will be lower case with _ to separate words. The classes will have capitalized words with no spaces. Directories will be capitalized with _ to separate words.

If the user would like to create test code please do this in the Test folder. Also this code is not be uploaded to the main code base on GitHub.

Workflow
========
The GeoData take advantage of a standardized structure of data to give the user access to the avalible tools. It's built off of container class where each instances is a specfic data set. In all cases the user needs to put their data in this structure. This first task will require a line of code similar to the following to start the process::

	Geo = GeoData(readfunction,input1,input2 ...)

The readfunction is a function that can read the data from its previous format to the one specified by GeoData. The terms input1, input2 are what ever inputs are required by the read function to work.

Once the data set is now in the proper format the user can go about augmenting it in a number of ways. The user can augment the values and labeling of the data sets by using the changedata method built into the class. Interpolation methods are avalible in the class to change the coordinate system or simply regrid it in the current system. The size of the data set can be reduced by applying methods to filter out specfic time and data points. A time registration method is also avalible where it will take as input a second instance of the class and determine what measurements overlap in time with the original instance.

At this point the user can plot their results. Each of the plotting tools are set up in different functions in the Plotting folder. These plotting tools will output handles to figures that we plotted along with handles to colorbars if included.


Having difficulty?
------------------

Fast 3-D plotting typically involves OpenGL these days.
Mayavi/VTK use OpenGL to make highly dense 3-D plots beautiful.
If you get `an OpenGL error like this <https://gist.github.com/scienceopen/da7f89e22ced7929c09f>`_ try::

	$ sudo apt-get install mayavi2
	$ /usr/bin/python2 mycode.py

where ``mycode.py`` is the file you want to run.
This uses your distribution's setup of Mayavi, which implicitly ought to be the most likely one to work!

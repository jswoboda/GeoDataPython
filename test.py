#!/usr/bin/env python
"""
Created on Wed Aug 20 12:34:52 2014

@author: Bodangles
"""

import inspect, os
print inspect.getfile(inspect.currentframe()) # script filename (usually with path)
print os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) #
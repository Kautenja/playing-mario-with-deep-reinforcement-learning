"""This module adjusts the PATH so scripts can run at the top level."""
import os as _os
from sys import path as _path
import matplotlib


# get the path to this directory
_THIS_DIR = _os.path.dirname(__file__)
# get the path to the parent module
_PARENT_MODULE = _os.path.abspath(_os.path.join(_THIS_DIR, _os.pardir))
# check if the parent path is already in the python PATH
if _PARENT_MODULE not in _path:
    # add the parent path to the python PATH
    _path.append(_PARENT_MODULE)


# set matplotlib to override default X11 environment
matplotlib.use('Agg')


# explicitly define the outward facing API of this module
__all__ = []

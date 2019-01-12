"""This module adjusts the PATH so scripts can run at the top level."""
import os as _os
from sys import path as _path
import matplotlib


def top_level(is_headless: bool=False):
    # get the path to this directory
    this_dir = _os.path.dirname(__file__)
    # get the path to the parent module
    parent_dir = _os.path.abspath(_os.path.join(this_dir, _os.pardir))
    # check if the parent path is already in the python PATH
    if parent_dir not in _path:
        # add the parent path to the python PATH
        _path.append(parent_dir)

    if is_headless:
        # set matplotlib to override default X11 environment
        matplotlib.use('Agg')


# explicitly define the outward facing API of this module
__all__ = [top_level.__name__]

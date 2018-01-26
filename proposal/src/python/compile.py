"""This module compiles the project by copying files to the build directory."""
import os
import shutil
from distutils.dir_util import copy_tree
from concatenate import concatenate_files

# source directories in the parent with the paper materials
SOURCE_DIRECTORIES = ['bib', 'sty', 'tex', 'bst']
# the default build directory
BUILD = './build'
# the directory this file is in
THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def path_in_parent(filename: str) -> str:
    """
    Return the path to the file in the parent directory.

    Args:
        filename: the filename in the parent directory

    Returns: the path to the filename
    """
    return '{}/../{}'.format(THIS_DIRECTORY, filename)


# if there is no build directory, make one
if not os.path.exists(BUILD):
    os.makedirs(BUILD)
# otherwise delete it and make a new one
else:
    shutil.rmtree(BUILD)
    os.makedirs(BUILD)


# copy the entire contents of the file tree to the build folder
[copy_tree(path_in_parent(src), BUILD) for src in SOURCE_DIRECTORIES]
# copy the images directory
copy_tree(path_in_parent('md/img'), BUILD + '/img')
# delete any READMEs from the build folder
if os.path.isfile(BUILD + '/README.md'):
    os.remove(BUILD + '/README.md')
# concatenate the markdown files into a build file
concatenate_files(path_in_parent('md'), BUILD + '/build.md')

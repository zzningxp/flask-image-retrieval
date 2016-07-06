import sys
import os
repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)

#support_dirname = os.path.join(repo_dirname, 'test', 'support')
temp_dirname = os.path.join(repo_dirname, 'test', '_temp')

import simplejson as json
import unittest
import numpy as np
from numpy.testing import *
from IPython import embed

import imagesearch
from imagesearch import image_caffe
from imagesearch import image_retrieval


#!/usr/bin/python

from __future__ import  absolute_import

import os
import sys, platform
from pprint import pprint

def tf_info ():
  import tensorflow as tf

  sysconfig = tf.sysconfig.get_build_info()
  info = {
    'tensorflow': tf.__version__,
    'python': platform.python_version(),
    'cuda': sysconfig["cuda_version"],
    'cudnn': sysconfig["cudnn_version"]
  }
  return info

pprint (tf_info ())


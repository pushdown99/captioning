from __future__ import  absolute_import
import os

import ipdb
import json
import time
import codecs
import matplotlib
import sys, platform
from tqdm import tqdm
from glob import glob
from os.path import join, basename
from lib.model import transformer
from lib.utils import Display
from lib.config import opt

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


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

def torch_info ():
  import torch
  info = {
    'torch': torch.__version__,
    'python': platform.python_version(),
    'cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version()
  }
  return info

def latest_model ():
  model = dict ()
  for d in glob('output/*_{}_*'.format(opt.data)):
    model[d] =  float(d.split('_')[5])
  return max (model, key=model.get)

def samples ():
  data = json.load(codecs.open(join(opt.data_dir, 'scores.json'), 'r', 'utf-8-sig'))
  files = {d['path']:d['bleu1']  for d in data}
  return max (files, key=files.get)

##########################################################
def eval (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  _model = opt.trained
  if _model == '': _model = latest_model ()
  _model = _model + '/' if _model[-1:] != '/' else _model
  model.evaluate(_model)

def inference (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  _model = opt.trained
  if _model == '': _model = latest_model ()
  _model = _model + '/' if _model[-1:] != '/' else _model

  _sample = opt.sample
  if _sample == '': _sample = samples ()
  model.inference(_sample, _model)
  Display (_sample)

def train (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  model.fit(opt.epoch)

##########################################################

if __name__ == '__main__':
  _t = time.process_time()
  print ('[+] Information: {}'.format(tf_info()))

  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))

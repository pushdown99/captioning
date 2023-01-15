from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm
from lib.model import transformer
from lib.utils import Display
from lib.config import opt

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def eval (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  _model = opt.trained
  _model = _model + '/' if _model[-1:] != '/' else _model
  model.evaluate(_model)

def inference (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  _model = opt.trained
  _model = _model + '/' if _model[-1:] != '/' else _model

  _sample = opt.sample
  model.inference(_sample, _model)
  Display (_sample)

def train (**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  model.fit(opt.epoch)

if __name__ == '__main__':
    import fire

    fire.Fire()

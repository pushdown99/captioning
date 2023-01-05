from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm
from lib.model import transformer
from lib.config import opt

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=10000):
  opt._parse(kwargs)

def train(**kwargs):
  opt._parse(kwargs)

  model = transformer.efficientnetb0() 
  model.fit(40)

if __name__ == '__main__':
    import fire

    fire.Fire()

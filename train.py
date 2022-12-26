import lib  # add lib folder to sys.path
import os
import sys
import time
import argparse
import tensorflow as tf
from colorama import init, deinit, Back, Fore
from config import cfg

from model import dataset, transformer

data   = dataset.nia(cfg)
#data   = dataset.coco(cfg)
model = transformer.efficientnetb0(cfg) 
model.fit(20) # epoch

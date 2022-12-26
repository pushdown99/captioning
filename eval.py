import lib  # add lib folder to sys.path
import os
import sys
import time
import argparse
import tensorflow as tf
from colorama import init, deinit, Back, Fore
from config import cfg

from model import dataset, transformer

#data   = dataset.nia(cfg)
model = transformer.efficientnetb0(cfg) 
model.evaluate('data/20221224_nia_e20_7_v40205_176255_22685_22832/')


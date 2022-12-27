import lib  # add lib folder to sys.path
import os
import sys
import time
import argparse
import tensorflow as tf
from colorama import init, deinit, Back, Fore
from config import cfg

from model import dataset, transformer

parser = argparse.ArgumentParser(description='Image Captioning (transformer)')
subparsers = parser.add_subparsers(dest='mode', help='main mode of network')
formatter = argparse.ArgumentDefaultsHelpFormatter

# create the parser for the train mode
parser_train = subparsers.add_parser('train', formatter_class=formatter, help='help for TRAIN mode of network')
parser_train.add_argument('-e', '--epoch',  type=int, default=20, help='number of epochs for training')
parser_train.add_argument('-n', '--name',   default='nia', help='dataset name for TRAIN mode of network: nia | coco')
parser_train.add_argument('-v', '--volume', default='full', help='data volume for TRAIN mode of network: full | fat | small')

# create the parser for the eval mode
parser_eval = subparsers.add_parser('eval', formatter_class=formatter, help='help for EVAL  mode of network')
parser_eval.add_argument('-p', '--path', help='path for EVAL mode of network')

def train(name, volume, epoch):
  if name == 'nia':
    data = dataset.nia(cfg, volume)

  if name == 'coco':
    data = dataset.coco(cfg, volume)

  model = transformer.efficientnetb0(cfg)
  model.fit(epoch)

  del model
  del data


def eval(m):
  model = transformer.efficientnetb0(cfg)
  model.evaluate(m)

  del model


if __name__ == "__main__":
  init(autoreset=True)

  args = parser.parse_args()
  if args.mode is None:
    parser.print_help()
    exit()

  print(Back.WHITE + Fore.BLACK + 'Called with args:')
  print(args)

  if args.mode == 'train':
      train (args.name, args.volume, args.epoch)

  elif args.mode == 'eval':
      eval (args.path)



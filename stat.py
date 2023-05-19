import  os, re, sys, time, math, json, random, codecs, argparse
import json
import pickle
import numpy
import string
import datetime

from os.path import basename, join, relpath, realpath, abspath, exists
from pprint import pprint
from glob import glob
from tqdm import tqdm
from pathlib import Path
from nltk import flatten

class Option:
  dataset    = 'nia'
  info_dir   = 'dataset'

opt = Option ()

image_json  = join(opt.info_dir, '_images.pkl' )
train_json  = join(opt.info_dir, 'c_train.json' )
val_json    = join(opt.info_dir, 'c_val.json' )
test_json   = join(opt.info_dir, 'c_test.json' )

def get_id_from_file (f):
  return basename(f).split('_')[0]+'_'+basename(f).split('_')[1]

def get_name_cls_ins (path):
  name = basename(path).split('.')[0]
  cls  = name.split('(')[0].split('_')[2:]
  cls = ' '.join(cls)
  ins  = name.split('(')[1].split(')')[0]

  return name, cls, ins

def info ():
  with open (image_json,  'rb') as fp: image  = pickle.load (fp)
  with open (train_json,  'rb') as fp: train  = json.load (fp)
  with open (val_json,  'rb') as fp: val  = json.load (fp)
  with open (test_json,  'rb') as fp: test  = json.load (fp)

  print ()

  print (current(), line2_80 ())
  print (current(), '{:15s} {:8s}'.format('TYPE', 'VALUE'))
  print (current(), line1_80 ())
  print (current(), '{:15s} {:8d}'.format('IMAGES', len(image)))
  n = len(train)+len(val)+len(test)
  print (current(), '{:15s} {:8d} {:5.1f}%'.format('DATASET',n, n*100/len(image)))
  print (current(), '{:15s} {:8d} {:5.1f}%'.format('- TRAIN',len(train), len(train)*100/n))
  print (current(), '{:15s} {:8d} {:5.1f}%'.format('- VAL',len(val), len(val)*100/n))
  print (current(), '{:15s} {:8d} {:5.1f}%'.format('- TEST',len(test), len(test)*100/n))
  print (current(), line2_80 ())

  stats = dict ()
  for k,v in image.items():
    name, cls, ins = get_name_cls_ins (v)

    if not cls in stats:
      stats[cls] = dict ()
      stats[cls]['total'] = 0
      stats[cls]['train'] = 0
      stats[cls]['val']   = 0
      stats[cls]['test']  = 0

    stats[cls]['total'] += 1

  for k,v in train.items():
    name, cls, ins = get_name_cls_ins (k)
    stats[cls]['train'] += 1

  for k,v in val.items():
    name, cls, ins = get_name_cls_ins (k)
    stats[cls]['val'] += 1

  for k,v in test.items():
    name, cls, ins = get_name_cls_ins (k)
    stats[cls]['test'] += 1

  print ()
  print (current(), line2_80 ())
  print (current(), '{:20s} {:10s} {:10s} {:10s} {:10s}'.format('CLASS', 'IMAGES', 'TRAIN', 'VAL', 'TEST'))
  print (current(), line1_80 ())
  for k,v in stats.items():
    if v['train'] <= 0:
      continue
    print (current(), '{:20s} {:10d} {:10d} {:10d} {:10d}'.format(k, v['total'], v['train'], v['val'], v['test']))
  print (current(), line2_80 ())
      

def current ():
  return  datetime.datetime.now()

def line1_80 ():
  return '--------------------------------------------------------------------------------'

def line2_80 ():
  return '================================================================================'

def start (argv):
  t = time.process_time()

  command = list()
  command.append ('python')
  command += argv

  print (current(), line1_80())
  print (current(), '[Run] $', ' '.join(command))
  print (current(), line1_80())

  return t

def stop (t):
  print ('')
  print (current(), line1_80())
  print (current(), '[End]')
  print (current(), line1_80())

  elapsed = time.process_time() - t
  print ('')
  print ('Total elapsed: {:.2f} sec'.format(elapsed))
  print ('')

if __name__ == '__main__':
  t = start (sys.argv)

  import fire
  fire.Fire()

  stop (t)


import lib
import os
import re
import sys
import time
import json
import random
import codecs
import argparse
from colorama import init, deinit, Back, Fore
from tqdm import tqdm
from glob import glob
from pickle import dump,load
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict

cfg = edict ()
cfg.ROOT_DIR   = 'data/GENOME'
cfg.DATA_DIR   = 'top_150_50.original'
cfg.TRAIN_FULL_JSON  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train.json'      )
cfg.TRAIN_FAT_JSON   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_fat.json'  )
cfg.TRAIN_SMALL_JSON = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_small.json')
cfg.TEST_FULL_JSON   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test.json'       )
cfg.TEST_FAT_JSON    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_fast.json'  )
cfg.TEST_SMALL_JSON  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_small.json' )

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed:', elapsed_time)

#o1 = json.load(codecs.open(cfg.TRAIN_FULL_JSON,  'r', 'utf-8-sig'))
o2 = json.load(codecs.open(cfg.TRAIN_FAT_JSON,   'r', 'utf-8-sig'))
#o3 = json.load(codecs.open(cfg.TRAIN_SMALL_JSON, 'r', 'utf-8-sig'))

objects    = dict()
predicates = dict()

for obj in o2:
  for o in obj['objects']:
   c = o['class']

   if not c in objects:
     objects[c] = 0
   objects[c] += 1

  for o in obj['relationships']:
   p = o['predicate']

   if not p in predicates:
     predicates[p] = 0
   predicates[p] += 1


objects    = {k: v for k, v in sorted(objects.items(), key=lambda item: item[1], reverse=True)}
predicates = {k: v for k, v in sorted(predicates.items(), key=lambda item: item[1], reverse=True)}

print (json.dumps(objects, indent=2, ensure_ascii=False))
print ('len:', len(objects))

print (json.dumps(predicates, indent=2, ensure_ascii=False))
print ('len:', len(predicates))


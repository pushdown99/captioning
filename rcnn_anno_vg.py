import lib
import os
import re
import sys
import time
import math
#import orjson as json
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
from collections import defaultdict

cfg = edict ()
cfg.ROOT     = 'data/GENOME'
cfg.DATA     = 'top_150_50'
cfg.IMAGEDIR = 'VG_100K'
cfg.TRAIN    = join(cfg.ROOT, cfg.DATA, 'train.json')
cfg.TEST     = join(cfg.ROOT, cfg.DATA, 'test.json' )
cfg.OBJECTS  = join(cfg.ROOT, cfg.DATA, 'categories.json' )

cfg._TRAINVAL   = join(cfg.ROOT, 'trainval.txt')
cfg._TEST       = join(cfg.ROOT, 'test.txt')
cfg.INSTANCE   = join(cfg.ROOT,  'instance.json')
cfg.IMAGES     = join(cfg.ROOT,  'images.json')
cfg.OBJECT_ID  = join(cfg.ROOT,  'object_id.json')
cfg.ID_OBJECT  = join(cfg.ROOT,  'id_object.json')

cfg.VOLUME     = 100000
cfg.TRAIN_RATE = 0.8
cfg.VAL_RATE   = 0.1
cfg.TEST_RATE  = 0.1


def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('elapsed:', elapsed_time)

images = {basename(p).split('.')[0]:p for p in tqdm(glob(join(cfg.ROOT, cfg.IMAGEDIR, '*.jpg'), recursive=True))}

object_id = dict()
obj = json.load(codecs.open(cfg.OBJECTS, 'r', 'utf-8-sig'))
for i, d in enumerate(obj['object']):
  object_id[d] = i
id_object = { str(id): object for (object, id) in object_id.items() }

instances = dict()
obj = json.load(codecs.open(cfg.TRAIN, 'r', 'utf-8-sig'))
for i, o in enumerate(obj):
  for d in o['objects']:
    box  = list(map(int, d['box'])) # x y w h
    name = d['class']
    path = o['path']
    id   = basename(path).split('.')[0]

    if not id in images:
      print ('id not found:', id)
      continue
    if not id in instances:
      instances[id] = list()

    inst = dict()
    bbox = [0 for i in range(4)]   # xmin ymin width height
    bbox[0] = int(box[1])          # ymin
    bbox[1] = int(box[0])          # xmin
    bbox[2] = int(box[1]+box[3])   # ymax = miny + height
    bbox[3] = int(box[0]+box[2])   # xmax = minx + width
    inst['bbox']  = bbox # ymin xmin ymax xmax
    inst['xywh']  = box
    inst['name']  = name
    inst['image'] = path #join(cfg.ROOT, cfg.IMAGEDIR, path)

    instances[id].append(inst)

#####################################################

data = [basename(v[0]['image']).split('.')[0] for v in list(instances.values())]

if cfg.VOLUME > len(data):
  cfg.VOLUME = len(data)

cfg.N_TRAIN    = int(cfg.VOLUME * cfg.TRAIN_RATE)
cfg.N_VAL      = int(cfg.N_TRAIN + cfg.VOLUME * cfg.VAL_RATE)
cfg.N_TRAINVAL = cfg.N_TRAIN + cfg.N_VAL
cfg.N_TEST     = int(cfg.N_VAL   + cfg.VOLUME * cfg.TEST_RATE)

print ('estimate:', cfg.VOLUME, cfg.N_TRAIN, cfg.N_VAL, cfg.N_TEST)

trainvals = data[:cfg.N_VAL]
tests     = data[cfg.N_VAL:cfg.VOLUME]

print ('volume   :', len(trainvals), len(tests))

fp = open(cfg._TRAINVAL, 'w')
fp.write('\n'.join(trainvals))
fp.close()

fp = open(cfg._TEST, 'w')
fp.write('\n'.join(tests))
fp.close()

json.dump(instances, open(cfg.INSTANCE,    'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(images,    open(cfg.IMAGES,      'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(object_id, open(cfg.OBJECT_ID,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(id_object, open(cfg.ID_OBJECT,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)

#VG_BBOX_LABEL_NAMES = tuple(json.load(codecs.open(cfg.OBJECT_ID, 'r', 'utf-8-sig')).keys())
#print (VG_BBOX_LABEL_NAMES)

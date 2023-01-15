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
cfg.ROOT     = 'data/COCO'
cfg.DATA     = 'annotations'
cfg.PREFIX   = '2017'
cfg.TRAIN    = join(cfg.ROOT, 'train'+cfg.PREFIX)
cfg.VAL      = join(cfg.ROOT, 'val'+cfg.PREFIX)
cfg.TEST     = join(cfg.ROOT, 'test'+cfg.PREFIX)
cfg.TRAIN_INFO     = join(cfg.ROOT, cfg.DATA, 'image_info_test-dev'+cfg.PREFIX+'.json')
cfg.TEST_INFO      = join(cfg.ROOT, cfg.DATA, 'image_info_test'+cfg.PREFIX+'.json')
cfg.TRAIN_INSTANCE = join(cfg.ROOT, cfg.DATA, 'instances_train'+cfg.PREFIX+'.json')
cfg.VAL_INSTANCE   = join(cfg.ROOT, cfg.DATA, 'instances_val'+cfg.PREFIX+'.json')
cfg.TEST_INSTANCE  = join(cfg.ROOT, cfg.DATA, 'instances_test'+cfg.PREFIX+'.json')

cfg._TRAINVAL   = join(cfg.ROOT, cfg.DATA, 'trainval.txt')
cfg._TEST       = join(cfg.ROOT, cfg.DATA, 'test.txt')
cfg.INSTANCE   = join(cfg.ROOT,  cfg.DATA, 'instance.json')
cfg.OBJECT_ID  = join(cfg.ROOT,  cfg.DATA, 'object_id.json')
cfg.ID_OBJECT  = join(cfg.ROOT,  cfg.DATA, 'id_object.json')

cfg.VOLUME     = 100000
cfg.TRAIN_RATE = 0.8
cfg.VAL_RATE   = 0.1
cfg.TEST_RATE  = 0.1


def round_up(n, decimals=0):
  multiplier = 10 ** decimals
  return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
  multiplier = 10 ** decimals
  return math.floor(n * multiplier) / multiplier

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('elapsed:', elapsed_time)

trains = {basename(p).split('.')[0]:p for p in tqdm(glob(join(cfg.TRAIN, '*.jpg'), recursive=True))}
valids = {basename(p).split('.')[0]:p for p in tqdm(glob(join(cfg.VAL,   '*.jpg'), recursive=True))}
tests  = {basename(p).split('.')[0]:p for p in tqdm(glob(join(cfg.TEST,  '*.jpg'), recursive=True))}
trainval = trains | valids

if cfg.VOLUME > len(trainval):
  cfg.VOLUME = len(trainval)

cfg.N_TRAIN    = int(cfg.VOLUME * cfg.TRAIN_RATE)
cfg.N_VAL      = int(cfg.N_TRAIN + cfg.VOLUME * cfg.VAL_RATE)
cfg.N_TRAINVAL = cfg.N_TRAIN + cfg.N_VAL
cfg.N_TEST     = int(cfg.N_VAL   + cfg.VOLUME * cfg.TEST_RATE)

print ('estimate:', cfg.VOLUME, cfg.N_TRAIN, cfg.N_VAL, cfg.N_TEST)

#trains = trains + valids
#random.shuffle (trains)

object_id = dict()

data = json.load(codecs.open(cfg.TRAIN_INFO, 'r', 'utf-8-sig')) # orjson.loads(codecs.open(cfg.TRAIN_INFO, 'r', 'utf-8-sig').read())
for d in data['categories']:
  id   = d['id']
  name = d['name']

  if not name in object_id:
    object_id[name] = int(id)

id_object = { str(id): object for (object, id) in object_id.items() }

t = time_start('loading', cfg.TRAIN_INSTANCE)
obj1 = json.load(codecs.open(cfg.TRAIN_INSTANCE, 'r', 'utf-8-sig'))
time_stop (t)

t = time_start('loading', cfg.VAL_INSTANCE)
obj2 = json.load(codecs.open(cfg.VAL_INSTANCE, 'r', 'utf-8-sig'))
time_stop (t)

obj = obj1['annotations'] + obj2['annotations']

data = defaultdict(list)
{data['{:012d}'.format(o['image_id'])].append(o) for o in obj}


trainval  = dict()
instances = dict()

for i, id in enumerate(trains):
  if not id in data:
#    print ('not found', id)
    continue

  if i >= cfg.VOLUME:
    break

  if not id in trainval:
    trainval[id] = 0
  trainval[id] += 1

  for d in data[id]:
    box  = list(map(int, d['bbox'])) # x y w h
    cid  = d['category_id']
    name = id_object[str(cid)]
  
    if not id in instances:
      instances[id] = list()

    inst = dict()
    inst['name'] = name
    bbox = [0 for i in range(4)]   # xmin ymin width height
    bbox[0] = int(box[1])          # ymin
    bbox[1] = int(box[0])          # xmin
    bbox[2] = int(box[1]+box[3])   # ymax = miny + height
    bbox[3] = int(box[0]+box[2])   # xmax = minx + width
    inst['bbox']  = bbox # ymin xmin ymax xmax
    inst['xywh']  = box
    inst['name']  = id_object[str(d['category_id'])]
    inst['image'] = trains[id]

    instances[id].append(inst)

#####################################################

data = [basename(v[0]['image']).split('.')[0] for v in list(instances.values())]

trainvals = data[:cfg.N_VAL]
tests     = data[cfg.N_VAL:cfg.VOLUME]

print ('volume   :', len(trainvals), len(tests))

fp = open(cfg._TRAINVAL, 'w')
fp.write('\n'.join(trainval))
fp.close()

fp = open(cfg._TEST, 'w')
fp.write('\n'.join(tests))
fp.close()

json.dump(instances, open(cfg.INSTANCE,    'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(object_id, open(cfg.OBJECT_ID,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(id_object, open(cfg.ID_OBJECT,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)

COCO_BBOX_LABEL_NAMES = tuple(json.load(codecs.open(cfg.OBJECT_ID, 'r', 'utf-8-sig')).keys())
print (COCO_BBOX_LABEL_NAMES)

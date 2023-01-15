import lib
import os
import re
import sys
import time
import math
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
cfg.ROOT      = 'data/NIA'
cfg.DATA      = 'annotations'
cfg.IMAGES    = join(cfg.ROOT, 'images')
cfg.JSONS     = join(cfg.ROOT, 'download/4-2')
cfg.OBJECTS   = join(cfg.ROOT, 'objects.json')
cfg.OBJECT_ID = join(cfg.ROOT, 'object_id.json')

cfg._TRAINVAL   = join(cfg.ROOT, 'trainval.txt')
cfg._TEST       = join(cfg.ROOT, 'test.txt')
cfg._INSTANCES  = join(cfg.ROOT, 'instance.json')
cfg._IMAGES     = join(cfg.ROOT, 'images.json')

cfg.VOLUME     = 800 # 14627 < x <= 14628 null []
cfg.TRAIN_RATE = 0.8
cfg.VAL_RATE   = 0.1
cfg.TEST_RATE  = 0.1

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed:', elapsed_time)

images = {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(cfg.IMAGES, '**/*.jpg'), recursive=True), desc='images')}

instances = dict()

id_object = json.load(codecs.open(cfg.OBJECTS, 'r', 'utf-8-sig'))
object_id = { object: str(id) for (id, object) in id_object.items() }

for i, p in enumerate(tqdm(glob(join(cfg.JSONS, '**/*.json'), recursive=True), desc='jsons ')):
  k = basename(p).split('.')[0].split('_')[0] + '_' +basename(p).split('.')[0].split('_')[1]
  if not k in images:
    continue

  if len(instances) >= cfg.VOLUME:
    break

  k1 = basename(images[k]).split('.')[0]
  dat = json.load(codecs.open(p, 'r', 'utf-8-sig'))

  if len(dat['annotations']) == 0:
    continue

  instances[k1] = list()

  for d in dat['annotations']:
    an = dict()
    an['bbox'] = [0, 0, 0, 0]
    an['bbox'][0] = int(d['bbox'][1])
    an['bbox'][1] = int(d['bbox'][0])
    an['bbox'][2] = int(d['bbox'][1]+d['bbox'][3])
    an['bbox'][3] = int(d['bbox'][0]+d['bbox'][2])

    an['xywh'] = [0, 0, 0, 0]
    an['xywh'][0] = int(d['bbox'][0])
    an['xywh'][1] = int(d['bbox'][1])
    an['xywh'][2] = int(d['bbox'][2])
    an['xywh'][3] = int(d['bbox'][3])

    if not str(d['category_id']) in id_object:
      print ('category_id (', d['category_id'], ') not found! ', p)
      continue
    an['name']  = id_object[str(d['category_id'])]
    an['path']  = p
    an['image'] = images[k]

    instances[k1].append(an)

print ('instances:', len(instances))
#k = list(instances.keys())[14628]
#print(k, instances[k])

# shuffle
keys =  list(instances.keys())
random.shuffle(keys)
instances = {key:instances[key] for key in keys}

#data = [[basename(v['image']).split('.')[0] for v in l ] for l in list(instances.values())[:cfg.VOLUME]]
#data = [x for sublist in data for x in sublist]
#print (json.dumps(data, indent=2, ensure_ascii=False))
data = [basename(v[0]['image']).split('.')[0] for k, v in instances.items()]

if cfg.VOLUME > len(instances):
  cfg.VOLUME = instances

cfg.N_TRAIN    = int(cfg.VOLUME * cfg.TRAIN_RATE)
cfg.N_VAL      = int(cfg.N_TRAIN + cfg.VOLUME * cfg.VAL_RATE)
cfg.N_TRAINVAL = cfg.N_TRAIN + cfg.N_VAL
cfg.N_TEST     = int(cfg.N_VAL   + cfg.VOLUME * cfg.TEST_RATE)

print ('estimate:', cfg.VOLUME, cfg.N_TRAIN, cfg.N_VAL, cfg.N_TEST)

trainvals = data[:cfg.N_VAL]
tests     = data[cfg.N_VAL:cfg.VOLUME]

print ('volume   :', len(trainvals), len(tests))

json.dump(images,    open(cfg._IMAGES,    'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(instances, open(cfg._INSTANCES, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
json.dump(object_id, open(cfg.OBJECT_ID,  'w', encoding='utf-8'), indent=2, ensure_ascii=False)

fp = open(cfg._TRAINVAL, 'w')
fp.write('\n'.join(trainvals))
fp.close()

fp = open(cfg._TEST, 'w')
fp.write('\n'.join(tests))
fp.close()

NIA_BBOX_LABEL_NAMES = tuple(json.load(codecs.open(cfg.OBJECT_ID, 'r', 'utf-8-sig')).keys())
print (NIA_BBOX_LABEL_NAMES)

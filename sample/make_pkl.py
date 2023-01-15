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
from _pickle import dump,load
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict


cfg = edict ()
cfg.ROOT_DIR    = 'data/NIA'
cfg.DATA_DIR    = 'annotations'
cfg.JSON41_DIR  = 'download/4-1'
cfg.JSON42_DIR  = 'download/4-2'
cfg.JSON43_DIR  = 'download/4-3'
cfg.JSON44_DIR  = 'download/4-4'
cfg.IMAGE_DIR   = 'images'
cfg.FAT_SIZE    = 1.0
cfg.NORMAL_SIZE = 0.5
cfg.SMALL_SIZE  = 0.05
cfg.TRAIN_SIZE  = 0.8
cfg.VALID_SIZE  = 0.1
cfg.TEST_SIZE   = 0.1
cfg.CAPTION     = 10
cfg.DSET_DIR    = 'data'
cfg.TEMP_DIR    = 'temp'
cfg.OBJ_JSON    = join(cfg.DSET_DIR, 'objects.json')
cfg.IMG_JSON    = join(cfg.TEMP_DIR, 'img.json')
cfg.N41_JSON    = join(cfg.TEMP_DIR, 'n41.json')
cfg.N42_JSON    = join(cfg.TEMP_DIR, 'n42.json')
cfg.N43_JSON    = join(cfg.TEMP_DIR, 'n43.json')
cfg.N44_JSON    = join(cfg.TEMP_DIR, 'n44.json')
cfg.IMG_PKL     = join(cfg.TEMP_DIR, 'img.pkl')
cfg.N41_PKL     = join(cfg.TEMP_DIR, 'n41.pkl')
cfg.N42_PKL     = join(cfg.TEMP_DIR, 'n42.pkl')
cfg.N43_PKL     = join(cfg.TEMP_DIR, 'n43.pkl')
cfg.N44_PKL     = join(cfg.TEMP_DIR, 'n44.pkl')
cfg.OBJECTS     = json.load(codecs.open(cfg.OBJ_JSON, 'r', 'utf-8-sig'))

def get_categories(f):
  obj = re.findall('\(([^)]+)', f)[0] 
  idx = obj.find('(')
  if idx >= 0:
    obj = obj[:idx]
  ids = basename(f).split('_')[0] + '_' + basename(f).split('_')[1]
  cls = basename(f).split('_')[2].split('(')[0]
  return ids, cls, obj

def build_caption(f, d):
  fp = open(f, 'w')
  for k in d:
    if len(d[k]['ko']) < cfg.CAPTION:
      continue

    idx  = 0
    caption = list()
    for ko in d[k]['ko']:
      ko = ko.replace('\t','')
      ko = ko.replace('\n','')
      ko = ko.replace('\r','')
      ko = ko.replace('..','.')
      ko = ko.strip()

      if len(ko) < 5:
        continue
        
      caption.append(ko)

    if len(caption) < cfg.CAPTION:
      continue

    caption = caption[:cfg.CAPTION]
    for i, c in enumerate(caption):
      fp.write ('{}#{}\t{}\n'.format(d[k]['base'], i, c))

  fp.close()


def build_dataset(f, d):
  fp = open(f, 'w')
  for k in d:
    if len(d[k]['ko']) < cfg.CAPTION:
      continue
    fp.write ('{}\n'.format(d[k]['base']))
  fp.close()

#######################################

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed:', elapsed_time)

def prepare_images ():
  u = 0

  if os.path.exists(cfg.IMG_PKL):
    t = time_start ('Loading', cfg.IMG_PKL)
    #images = json.load(codecs.open(cfg.IMG_JSON, 'r', 'utf-8-sig'))
    images = load(open(cfg.IMG_PKL, 'rb'))
    time_stop (t)
  else:
    images  = dict ()

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.IMAGE_DIR, '*.jpg'), recursive=True), desc='Images'):
    ids, cls, obj = get_categories (p)

    if ids in images:
      continue

    images[ids] = p     # id -> path
    u += 1


  if u > 0:
    t = time_start ('Saving ', cfg.IMG_PKL)
    #json.dump(images, open(cfg.IMG_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(images, open(cfg.IMG_PKL, 'wb'))
    time_stop (t)

def prepare_nia_4_2 ():
  u = 0

  if os.path.exists(cfg.N42_PKL):
    t = time_start ('Loading', cfg.N42_PKL)
    #nia = json.load(codecs.open(cfg.N42_JSON, 'r', 'utf-8-sig'))
    nia = load(open(cfg.N42_PKL, 'rb'))
    time_stop (t)
  else:
    nia  = dict ()

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON42_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if ids in nia:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))

      data = dict ()
      data['id']      = object['images'][0]['id']
      data['width']   = object['images'][0]['width']
      data['height']  = object['images'][0]['height']
      data['objects'] = list()

      for b in object['annotations']:
        d = dict ()
        d['id']    = b['category_id']
        d['class'] = cfg.OBJECTS[str(b['category_id'])]
        d['box']   = list(map(int, b['bbox']))
        data['objects'].append(d)

      nia[ids] = data     
      u += 1

    except error as e:
      print (p)
      continue

  if u > 0:
    t = time_start ('Saving ', cfg.N42_PKL)
    #json.dump(nia, open(cfg.N42_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(nia, open(cfg.N42_PKL, 'wb'))
    time_stop (t)

def prepare_nia_4_3 ():
  u = 0
  if os.path.exists(cfg.N43_PKL):
    t = time_start ('Loading', cfg.N43_PKL)
    #nia = json.load(codecs.open(cfg.N43_JSON, 'r', 'utf-8-sig'))
    nia = load(open(cfg.N43_PKL, 'rb'))
    time_stop (t)
  else:
    nia  = dict ()


  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON43_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if ids in nia:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))

      data = dict ()
      data['id']      = object['images'][0]['id']
      data['width']   = object['images'][0]['width']
      data['height']  = object['images'][0]['height']
      data['text']    = object['annotations'][0]['text']

      nia[ids] = data
      u += 1

    except error as e:
      print (p, e)
      continue

  if u > 0:
    t = time_start ('Saving ', cfg.N43_PKL)
    #json.dump(nia, open(cfg.N43_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(nia, open(cfg.N43_PKL, 'wb'))
    time_stop (t)

def prepare_nia_4_4 ():
  u = 0
  if os.path.exists(cfg.N44_PKL):
    t = time_start ('Loading', cfg.N44_PKL)
    #nia = json.load(codecs.open(cfg.N44_JSON, 'r', 'utf-8-sig'))
    nia = load(open(cfg.N44_PKL, 'rb'))
    time_stop (t)
  else:
    nia  = dict ()


  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON44_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if ids in nia:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))

      data = dict ()
      data['id']      = object['images']['id']
      data['width']   = object['images']['width']
      data['height']  = object['images']['height']
      data['text']    = object['annotations'][0]['text']

      nia[ids] = data
      u += 1

    except error as e:
      print (p, e)
      continue

  if u > 0:
    t = time_start ('Saving ', cfg.N44_PKL)
    #json.dump(nia, open(cfg.N44_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(nia, open(cfg.N44_PKL, 'wb'))
    time_stop (t)


def directory():
  path = 'data/GENOME/top_150_50/*.json'
  for p in glob(path):
    print (p)

def _inverse_weight():
  path = 'data/GENOME/top_150_50/inverse_weight.json'
  obj  = json.load(codecs.open(path, 'r', 'utf-8-sig'))
  print (json.dumps(obj, indent=2, ensure_ascii=False))

def _dict():
  path = 'data/GENOME/top_150_50/dict.json'
  obj  = json.load(codecs.open(path, 'r', 'utf-8-sig'))
  #print (json.dumps(obj, indent=2, ensure_ascii=False))
  print ('idx2word:', len(obj['idx2word']))
  print ('word2idx:', len(obj['word2idx']))

cfg.PRED_DIR  = 'download/4-4'

def predicate():
  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON_DIR, '**/*.json'), recursive=True)):
    print (p)

#print (cfg.OBJECTS)
#print (cfg.OBJECTS['1'])

prepare_images ()
prepare_nia_4_2 ()
prepare_nia_4_3 ()
prepare_nia_4_4 ()







# data/GENOME/top_150_50/inverse_weight.json
#_inverse_weight()
# data/GENOME/top_150_50/test_small.json
# data/GENOME/top_150_50/test.json
# data/GENOME/top_150_50/train.json
# data/GENOME/top_150_50/dict.json
#_dict()
# data/GENOME/top_150_50/train_fat.json
# data/GENOME/top_150_50/kmeans_anchors.json
# data/GENOME/top_150_50/categories.json
# data/GENOME/top_150_50/train_small.json


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
cfg.OBJ_JSON    = 'data/objects.json'
cfg.IMG_JSON    = 'data/img.json'
cfg.N41_JSON    = 'data/n41.json'
cfg.N42_JSON    = 'data/n42.json'
cfg.N43_JSON    = 'data/n43.json'
cfg.N44_JSON    = 'data/n44.json'
cfg.N41_PKL     = 'data/n41.pkl'
cfg.N42_PKL     = 'data/n42.pkl'
cfg.N43_PKL     = 'data/n43.pkl'
cfg.N44_PKL     = 'data/n44.pkl'
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
  images  = dict ()

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.IMAGE_DIR, '*.jpg'), recursive=True), desc='Images'):
    ids, cls, obj = get_categories (p)
    images[ids] = p     # id -> path

  json.dump(images, open(cfg.IMG_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

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


def prepare():
  images  = dict ()
  classes = dict ()
  regions = dict ()

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.IMAGE_DIR, '*.jpg'), recursive=True), desc='Images'):
    ids, cls, obj = get_categories (p)
    images[ids] = p     # id -> path

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if not ids in images:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))

      data = dict ()
      data['id']            = object['images']['id']
      data['width']         = object['images']['width']
      data['height']        = object['images']['height']
      data['path']          = basename(images[ids])
      data['relationships'] = list ()
      data['regions']       = list ()

      for b in object['annotations'][0]['text']:
        d = dict ()
        d['box'] = [0, 0, data['width'], data['height']]
        d['phase'] = b['korean'].split()
        data['regions'].append(d)

      if not ids in regions:
        regions[ids] = data

    except error as e:
      continue
    

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.BBOX_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if not ids in regions:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))

      regions[ids]['objects']       = list ()

      for b in object['annotations']:
        d = dict ()
        d['id']    = b['category_id']
        d['class'] = cfg.OBJECTS[str(b['category_id'])]
        d['box']   = list(map(int, b['bbox']))
        regions[ids]['objects'].append(d)

    except error as e:
      continue
    
    break

  print (regions)

#  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON_DIR, '**/*.json'), recursive=True), desc='Jsons '):
#    ids, cls, obj = get_categories (p)
#
#    if not ids in images:
#      continue
#
#    try:
#      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))
#      for c in object['annotations'][0]['text']:
#        if not ids in caption:
#          caption[ids] = dict ()
#          caption[ids]['ko']   = list ()
#          caption[ids]['en']   = list ()
#          caption[ids]['json'] = p
#          caption[ids]['path'] = images[ids]
#          caption[ids]['base'] = basename(images[ids])
#
#        caption[ids]['ko'].append(c['korean'] )
#        caption[ids]['en'].append(c['english'])
#
#    except error as e:
#      continue
##
#    if not cls in classes:
#      classes[cls] = dict ()
#      classes[cls]['ids'] = list ()
#      classes[cls]['num'] = 0
#
#    classes[cls]['ids'].append (ids)
#
#  for k in classes:
#    random.shuffle (classes[k]['ids'])
#    classes[k]['num'] = len(classes[k]['ids'])
#
#  for k in caption:
#    random.shuffle (caption[k]['ko'])
#    random.shuffle (caption[k]['en'])
#
#  json.dump(classes, open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_classes.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
#  json.dump(caption, open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_caption.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def build():
  classes = json.load(codecs.open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_classes.json'), 'r', 'utf-8-sig'))
  caption = json.load(codecs.open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_caption.json'), 'r', 'utf-8-sig'))

  total_fat    = dict()
  total_normal = dict()
  total_small  = dict()

  train_fat    = dict()
  train_normal = dict()
  train_small  = dict()

  valid_fat    = dict()
  valid_normal = dict()
  valid_small  = dict()
  test_fat     = dict()
  test_normal  = dict()
  test_small   = dict()

  for k in classes:
    n_total_fat    = int(len(classes[k]['ids']) * cfg.FAT_SIZE)
    n_train_fat    = int(n_total_fat * cfg.TRAIN_SIZE)
    n_valid_fat    = n_train_fat + int(n_total_fat * cfg.VALID_SIZE)
    n_test_fat     = n_train_fat + n_valid_fat + int(n_total_fat * cfg.TEST_SIZE)

    n_total_normal = int(n_total_fat * cfg.NORMAL_SIZE)
    n_train_normal = int(n_total_normal * cfg.TRAIN_SIZE)
    n_valid_normal = n_train_normal + int(n_total_normal * cfg.VALID_SIZE)
    n_test_normal  = n_train_normal + n_valid_normal + int(n_total_normal * cfg.TEST_SIZE)

    n_total_small  = int(n_total_fat * cfg.SMALL_SIZE)
    n_train_small  = int(n_total_small * cfg.TRAIN_SIZE)
    n_valid_small  = n_train_small + int(n_total_small * cfg.VALID_SIZE)
    n_test_small   = n_train_small + n_valid_small + int(n_total_small * cfg.TEST_SIZE)

    cnt = 0
    for ids in classes[k]['ids']:
      if cnt >= 0 and cnt < n_total_fat:
        total_fat[ids] = caption[ids]
      if cnt >= 0 and cnt < n_total_normal:
        total_normal[ids] = caption[ids]
      if cnt >= 0 and cnt < n_total_small:
        total_small[ids] = caption[ids]

      if cnt >= 0 and cnt < n_train_fat:
        train_fat[ids] = caption[ids]
      if cnt >= 0 and cnt < n_train_normal:
        train_normal[ids] = caption[ids]
      if cnt >= 0 and cnt < n_train_small:
        train_small[ids] = caption[ids]

      if cnt >= n_train_fat    and cnt < n_valid_fat:
        valid_fat[ids] = caption[ids]
      if cnt >= n_train_normal and cnt < n_valid_normal:
        valid_normal[ids] = caption[ids]
      if cnt >= n_train_small  and cnt < n_valid_small:
        valid_small[ids] = caption[ids]
  
      if cnt >= n_valid_fat    and cnt < n_test_fat:
        test_fat[ids] = caption[ids]
      if cnt >= n_valid_normal and cnt < n_test_normal:
        test_normal[ids] = caption[ids]
      if cnt >= n_valid_small  and cnt < n_test_small:
        test_small[ids] = caption[ids]

      cnt += 1

  f_total_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_fat.txt'   )
  f_total_normal = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_normal.txt')
  f_total_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_small.txt' )
  f_train_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_fat.txt'     )
  f_train_normal = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_normal.txt'  )
  f_train_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_small.txt'   )
  f_valid_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_fat.txt'     )
  f_valid_normal = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_normal.txt'  )
  f_valid_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_small.txt'   )
  f_test_fat     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_fat.txt'      )
  f_test_normal  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_normal.txt'   )
  f_test_small   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_small.txt'    )

  build_caption (f_total_fat,    total_fat   )
  build_caption (f_total_normal, total_normal)
  build_caption (f_total_small,  total_small )

  build_dataset (f_train_fat,    train_fat   )
  build_dataset (f_train_normal, train_normal)
  build_dataset (f_train_small,  train_small )

  build_dataset (f_valid_fat,    valid_fat   )
  build_dataset (f_valid_normal, valid_normal)
  build_dataset (f_valid_small,  valid_small )

  build_dataset (f_test_fat,     test_fat    )
  build_dataset (f_test_normal,  test_normal )
  build_dataset (f_test_small,   test_small  )

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
print (cfg.OBJECTS['1'])

#prepare_images ()
#prepare_nia_4_2 ()
#prepare_nia_4_3 ()
prepare_nia_4_4 ()
#prepare ()
#build ()
#directory()
#predicate()
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


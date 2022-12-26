import lib
import os
import gc
import re
import sys
import time
import json
import random
import codecs
import argparse
from colorama import init, deinit, Back, Fore
from langdetect import detect
from tqdm import tqdm
from glob import glob
from _pickle import dump,load
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict

cfg = edict ()
cfg.ROOT_DIR            = 'data/GENOME'
cfg.DATA_DIR            = 'top_150_50.original'
cfg.OUTPUT_DIR          = 'top_150_50'
cfg.REDUCE_RATE         = 0.1

cfg.TRAIN_IN        = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train.json')
cfg.TEST_IN         = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test.json' )
cfg.TRAIN_FAT_IN    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_fat.json')
cfg.TEST_FAT_IN     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_fat.json' )
cfg.TRAIN_SMALL_IN  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_small.json')
cfg.TEST_SMALL_IN   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_small.json' )

cfg.TRAIN_OUT       = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'train.json')
cfg.TEST_OUT        = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'test.json' )
cfg.TRAIN_FAT_OUT   = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'train_fat.json')
cfg.TEST_FAT_OUT    = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'test_fat.json' )
cfg.TRAIN_SMALL_OUT = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'train_small.json')
cfg.TEST_SMALL_OUT  = join(cfg.ROOT_DIR, cfg.OUTPUT_DIR, 'test_small.json' )

#######################################

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed: {0:.2f}'.format(elapsed_time))

#######################################

def reduce (f):
  data = list()

  obj = json.load(codecs.open(f, 'r', 'utf-8-sig'))
  length = len(obj)
  limit  = int(cfg.REDUCE_RATE * len(obj))

  print (f, length, ' -> ', limit)

  for idx, d in enumerate(obj):
    data.append(d)
    if idx == limit:
      break
  return data

def prepare():
  d = reduce (cfg.TRAIN_IN)
  json.dump(d, open(cfg.TRAIN_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

  d = reduce (cfg.TRAIN_SMALL_IN)
  json.dump(d, open(cfg.TRAIN_SMALL_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

  d = reduce (cfg.TEST_IN)
  json.dump(d, open(cfg.TEST_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

  d = reduce (cfg.TEST_SMALL_IN)
  json.dump(d, open(cfg.TEST_SMALL_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


'''
  u = 0
  genome     = dict ()
  objects    = dict ()
  predicates = dict ()
  words      = dict ()
  alias      = json.load(codecs.open(cfg.ALIAS_JSON, 'r', 'utf-8-sig'))

  gc.disable()
  if os.path.exists(cfg.IMG_PKL):
    t = time_start ('Loading', cfg.IMG_PKL)
    images = load(open(cfg.IMG_PKL, 'rb'))
    time_stop (t)
  else:
    images  = dict ()

  if os.path.exists(cfg.N42_PKL):
    t = time_start ('Loading', cfg.N42_PKL)
    nia42 = load(open(cfg.N42_PKL, 'rb'))
    time_stop (t)
  else:
    nia42  = dict ()

  if os.path.exists(cfg.N44_PKL):
    t = time_start ('Loading', cfg.N44_PKL)
    nia44 = load(open(cfg.N44_PKL, 'rb'))
    time_stop (t)
  else:
    nia44  = dict ()

  gc.enable()

  for ids in nia44.keys():
    if not ids in images:
      continue

    if not ids in nia42:
      continue

    data = dict()
    data['id']     = nia44[ids]['id']
    data['path']   = images[ids]
    data['base']   = basename(images[ids])
    data['height'] = nia44[ids]['height']
    data['width']  = nia44[ids]['width']

    data['relationships'] = list()
    for c in nia44[ids]['text']:
      d = dict ()
      d['sub_id']    = 1
      d['obj_id']    = 1
      d['entity1']   = c['entity1']
      d['entity2']   = c['entity2']
      d['verb']      = c['verb']
      d['predicate'] = c['relation'].lower()
      data['relationships'].append(d)

      #########################################

      k = d['entity1']
      if k is not None and len(k) > 0:
        k = k.strip()
        if k in alias:
          k = alias[k]
        if not k in objects:
          objects[k] = 0
        objects[k] += 1


      k = d['entity2']
      if k is not None and len(k) > 0:
        k = k.strip()
        if k in alias:
          k = alias[k]
        if not k in objects:
          objects[k] = 0
        objects[k] += 1

      k = d['predicate']
      if k is not None and len(k) > 0:
        k = k.strip()
        if not k in predicates:
          predicates[k] = 0
        predicates[k] += 1

    data['regions'] = list()
    for c in nia44[ids]['text']:
      d = dict ()
      d['box']    = [0, 0, data['width'], data['height']]
      d['phrase'] = c['korean']
      data['regions'].append(d)

      for p in d['phrase']:
        if not p in words:
          words[p] = 0
        words[p] += 1

    data['objects'] = list()
    for c in nia42[ids]['objects']:
      d = dict ()
      d['id']    = c['id']
      d['class'] = c['class']
      d['box']   = c['box']
      data['objects'].append(d)

    genome[ids] = data

  objects  = {k: v for k, v in sorted(objects.items(), key=lambda item: item[1], reverse=True)}

  objects1 = dict ()
  for k in objects:
    try:
      loc = detect(k)
      if loc == 'ko':
        continue
      objects1[k] = objects[k]
      if not k in cfg.OBJECTS:
        print ('not found', k)

    except Exception as err:
      print ('Exception: ', k, len(k), err)

  predicates = {k: v for k, v in sorted(predicates.items(), key=lambda item: item[1], reverse=True)}
  words      = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}

  categories = dict ()
  categories['predicate'] = list(predicates.keys())
  categories['object']    = list(objects.keys())

  categories1 = dict ()
  categories1['predicate'] = list(predicates.keys())
  categories1['object']    = list(objects1.keys())

  gc.disable()
  t = time_start ('Saving ', cfg.GENOME_JSON)
  json.dump(genome,      open(cfg.GENOME_JSON, 'w',       encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(objects,     open(cfg.OBJECTS_JSON, 'w',      encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(objects1,    open(cfg.OBJECTS1_JSON, 'w',     encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(predicates,  open(cfg.PREDICATES_JSON, 'w',   encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(words,       open(cfg.WORDS_JSON, 'w',        encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(categories,  open(cfg.CATEGOTIES_JSON, 'w',   encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(categories1, open(cfg.CATEGOTIES1_JSON, 'w',  encoding='utf-8'), indent=2, ensure_ascii=False)
  time_stop (t)
  gc.enable()

def build():
  classes = json.load(codecs.open(cfg.CLASS_JSON  , 'r', 'utf-8-sig'))
  caption = json.load(codecs.open(cfg.CAPTION_JSON, 'r', 'utf-8-sig'))

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
    n_total_normal = int(len(classes[k]['ids']) * cfg.NORMAL_SIZE)
    n_train_normal = int(n_total_normal * cfg.TRAIN_SIZE)
    n_valid_normal = n_train_normal + int(n_total_normal * cfg.VALID_SIZE)
    n_test_normal  = n_train_normal + n_valid_normal + int(n_total_normal * cfg.TEST_SIZE)

    n_total_fat    = int(n_total_normal * cfg.FAT_SIZE)
    n_train_fat    = int(n_total_fat * cfg.TRAIN_SIZE)
    n_valid_fat    = n_train_fat + int(n_total_fat * cfg.VALID_SIZE)
    n_test_fat     = n_train_fat + n_valid_fat + int(n_total_fat * cfg.TEST_SIZE)

    n_total_small  = int(n_total_normal * cfg.SMALL_SIZE)
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

prepare ()
#build ()
'''

prepare ()


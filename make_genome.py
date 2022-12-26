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
#from pickle import dump,load
from _pickle import dump,load
#import _pickle as cPickle
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict

cfg = edict ()
cfg.ROOT_DIR    = 'data/NIA'
cfg.DATA_DIR    = 'annotations'
cfg.JSON_DIR    = join(cfg.ROOT_DIR, 'download/4-3')
cfg.NORMAL_SIZE = 1.0
cfg.FAT_SIZE    = 0.5
cfg.SMALL_SIZE  = 0.05
cfg.TRAIN_SIZE  = 0.8
cfg.VALID_SIZE  = 0.1
cfg.TEST_SIZE   = 0.1
cfg.CAPTION     = 10
cfg.OBJECTS     =  json.load(codecs.open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'objects.json'), 'r', 'utf-8-sig'))
cfg.OBJECTS     = {v: k for k, v in cfg.OBJECTS.items()}

cfg.IMG_PKL     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'img.pkl')
cfg.N42_PKL     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'n42.pkl')
cfg.N44_PKL     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'n44.pkl')
cfg.CLASS_PKL   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_classes.pkl')
cfg.CAPTION_PKL = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_caption.pkl')

cfg.ALIAS_JSON       = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'alias.json')
cfg.GENOME_JSON      = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'genome.json')
cfg.OBJECTS_JSON     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'objects_meta.json')
cfg.OBJECTS1_JSON    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'objects1_meta.json')
cfg.PREDICATES_JSON  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'predicates_meta.json')
cfg.WORDS_JSON       = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'words_meta.json')
cfg.CATEGOTIES_JSON  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'categories_meta.json')
cfg.CATEGOTIES1_JSON = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'categories1_meta.json')

#######################################

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
  print ('Elapsed: {0:.2f}'.format(elapsed_time))

#######################################

def prepare():
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

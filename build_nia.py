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
cfg.ROOT_DIR     = 'data/NIA'
cfg.DATA_DIR     = 'annotations'
cfg.JSON_DIR     = join(cfg.ROOT_DIR, 'download/4-3')
cfg.TEMP_DIR     = 'temp'
cfg.CORPUS_JSON  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'corpus.json')
cfg.NORMAL_SIZE  = 1.0
cfg.FAT_SIZE     = 0.5
cfg.SMALL_SIZE   = 0.05
cfg.TRAIN_SIZE   = 0.8
cfg.VALID_SIZE   = 0.1
cfg.TEST_SIZE    = 0.1
cfg.CAPTION      = 10

cfg.IMG_JSON     = join(cfg.TEMP_DIR, 'img.json')
cfg.IMG_PKL      = join(cfg.TEMP_DIR, 'img.pkl')
cfg.N43_JSON     = join(cfg.TEMP_DIR, 'n43.json')
cfg.N43_PKL      = join(cfg.TEMP_DIR, 'n43.pkl')
cfg.CLASS_JSON   = join(cfg.TEMP_DIR, 'full_classes.json')
cfg.CLASS_PKL    = join(cfg.TEMP_DIR, 'full_classes.pkl')
cfg.CAPTION_JSON = join(cfg.TEMP_DIR, 'full_caption.json')
cfg.CAPTION_PKL  = join(cfg.TEMP_DIR, 'full_caption.pkl')

corpus = json.load(codecs.open(cfg.CORPUS_JSON,  'r', 'utf-8-sig'))

def get_categories(f):
  obj = re.findall('\(([^)]+)', f)[0] 
  idx = obj.find('(')
  if idx >= 0:
    obj = obj[:idx]
  ids = basename(f).split('_')[0] + '_' + basename(f).split('_')[1]
  cls = basename(f).split('_')[2].split('(')[0]
  return ids, cls, obj

def get_reduce (t):
  for k, v in corpus.items():
    t = t.replace(k, v)
  return t

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
        
#      spelled_sent = spell_checker.check(ko)
#      ko = spelled_sent.checked
      c = get_reduce (ko)
      caption.append(c)

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

def prepare():
  u = 0

  if os.path.exists(cfg.IMG_PKL):
    t = time_start ('Loading', cfg.IMG_PKL)
    images = load(open(cfg.IMG_PKL, 'rb'))
    time_stop (t)
  else:
    images  = dict ()

  if os.path.exists(cfg.N43_PKL):
    t = time_start ('Loading', cfg.N43_PKL)
    nia = load(open(cfg.N43_PKL, 'rb'))
    time_stop (t)
  else:
    nia  = dict ()

  classes = dict ()
  caption = dict ()

  for p in tqdm(glob(join(cfg.JSON_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if not ids in images:
      continue

    if not ids in nia:
      continue

    if not ids in caption:
      caption[ids] = dict ()
      caption[ids]['ko']   = list ()
      caption[ids]['en']   = list ()
      caption[ids]['json'] = p
      caption[ids]['path'] = images[ids]
      caption[ids]['base'] = basename(images[ids])
      for c in nia[ids]['text']:
        caption[ids]['ko'].append(c['korean'] )
        caption[ids]['en'].append(c['english'])

      u += 1

    if not cls in classes:
      classes[cls] = dict ()
      classes[cls]['ids'] = list ()
      classes[cls]['num'] = 0

    classes[cls]['ids'].append (ids)

  for k in classes:
    #random.shuffle (classes[k]['ids'])
    classes[k]['num'] = len(classes[k]['ids'])

  #for k in caption:
  #  random.shuffle (caption[k]['ko'])
  #  random.shuffle (caption[k]['en'])

  if u > 0:
    t = time_start ('Saving ', cfg.CLASS_JSON)
    json.dump(classes, open(cfg.CLASS_JSON,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(classes, open(cfg.CLASS_PKL, 'wb'))
    time_stop (t)

    t = time_start ('Saving ', cfg.CAPTION_JSON)
    json.dump(caption, open(cfg.CAPTION_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    dump(caption, open(cfg.CAPTION_PKL, 'wb'))
    time_stop (t)

def build():
  classes = json.load(codecs.open(cfg.CLASS_JSON  , 'r', 'utf-8-sig'))
  caption = json.load(codecs.open(cfg.CAPTION_JSON, 'r', 'utf-8-sig'))

  total_full   = dict()
  total_fat    = dict()
  total_small  = dict()

  train_full   = dict()
  train_fat    = dict()
  train_small  = dict()

  valid_full   = dict()
  valid_fat    = dict()
  valid_small  = dict()

  test_full    = dict()
  test_fat     = dict()
  test_small   = dict()

  for k in classes:
    n_total_full   = int(len(classes[k]['ids']) * cfg.NORMAL_SIZE)
    n_train_full   = int(n_total_full * cfg.TRAIN_SIZE)
    n_valid_full   = n_train_full + int(n_total_full * cfg.VALID_SIZE)
    n_test_full    = n_valid_full + int(n_total_full * cfg.TEST_SIZE)

    n_total_fat    = int(n_total_full * cfg.FAT_SIZE)
    n_train_fat    = int(n_total_fat * cfg.TRAIN_SIZE)
    n_valid_fat    = n_train_fat + int(n_total_fat * cfg.VALID_SIZE)
    n_test_fat     = n_valid_fat + int(n_total_fat * cfg.TEST_SIZE)

    n_total_small  = int(n_total_full * cfg.SMALL_SIZE)
    n_train_small  = int(n_total_small * cfg.TRAIN_SIZE)
    n_valid_small  = n_train_small + int(n_total_small * cfg.VALID_SIZE)
    n_test_small   = n_valid_small + int(n_total_small * cfg.TEST_SIZE)

    for i, ids in enumerate(classes[k]['ids']):
      if i >= 0 and i < n_total_full:
        total_full[ids] = caption[ids]
      if i >= 0 and i < n_train_full:
        train_full[ids] = caption[ids]
      if i >= n_train_full and i < n_valid_full:
        valid_full[ids] = caption[ids]
      if i >= n_valid_full and i < n_test_full:
        test_full[ids] = caption[ids]

      if i >= 0 and i < n_total_fat:
        total_fat[ids] = caption[ids]
      if i >= 0 and i < n_train_fat:
        train_fat[ids] = caption[ids]
      if i >= n_train_fat    and i < n_valid_fat:
        valid_fat[ids] = caption[ids]
      if i >= n_valid_fat    and i < n_test_fat:
        test_fat[ids] = caption[ids]

      if i >= 0 and i < n_total_small:
        total_small[ids] = caption[ids]
      if i >= 0 and i < n_train_small:
        train_small[ids] = caption[ids]
      if i >= n_train_small  and i < n_valid_small:
        valid_small[ids] = caption[ids]
      if i >= n_valid_small  and i < n_test_small:
        test_small[ids] = caption[ids]

  print('full :', len(total_full), len(train_full), len(valid_full), len(test_full))
  print('fat  :', len(total_fat), len(train_fat), len(valid_fat), len(test_fat))
  print('small:', len(total_small), len(train_small), len(valid_small), len(test_small))

  f_total_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_full.txt' )
  f_total_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_fat.txt'  )
  f_total_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption_small.txt')

  f_train_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_full.txt'   )
  f_train_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_fat.txt'    )
  f_train_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train_small.txt'  )

  f_valid_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_full.txt'   )
  f_valid_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_fat.txt'    )
  f_valid_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid_small.txt'  )

  f_test_full    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_full.txt'    )
  f_test_fat     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_fat.txt'     )
  f_test_small   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test_small.txt'   )

  build_caption (f_total_fat,   total_fat   )
  build_caption (f_total_full,  total_full)
  build_caption (f_total_small, total_small )

  build_dataset (f_train_fat,   train_fat   )
  build_dataset (f_train_full,  train_full)
  build_dataset (f_train_small, train_small )

  build_dataset (f_valid_fat,   valid_fat   )
  build_dataset (f_valid_full,  valid_full)
  build_dataset (f_valid_small, valid_small )

  build_dataset (f_test_fat,   test_fat    )
  build_dataset (f_test_full,  test_full )
  build_dataset (f_test_small, test_small  )

prepare ()
build ()

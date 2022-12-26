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
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict

cfg = edict ()
cfg.ROOT_DIR  = 'data/NIA'
cfg.DATA_DIR  = 'annotations'
cfg.JSON_DIR  = 'download/4-3'
cfg.IMAGE_DIR = 'images'
cfg.FAT_SIZE    = 1.0
cfg.NORMAL_SIZE = 0.5
cfg.SMALL_SIZE  = 0.05
cfg.TRAIN_SIZE  = 0.8
cfg.VALID_SIZE  = 0.1
cfg.TEST_SIZE   = 0.1
cfg.CAPTION     = 10

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

def prepare():
  images  = dict ()
  classes = dict ()
  caption = dict ()

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.IMAGE_DIR, '*.jpg'), recursive=True), desc='Images'):
    ids, cls, obj = get_categories (p)
    images[ids] = p     # id -> path

  for p in tqdm(glob(join(cfg.ROOT_DIR, cfg.JSON_DIR, '**/*.json'), recursive=True), desc='Jsons '):
    ids, cls, obj = get_categories (p)

    if not ids in images:
      continue

    try:
      object = json.load(codecs.open(p, 'r', 'utf-8-sig'))
      for c in object['annotations'][0]['text']:
        if not ids in caption:
          caption[ids] = dict ()
          caption[ids]['ko']   = list ()
          caption[ids]['en']   = list ()
          caption[ids]['json'] = p
          caption[ids]['path'] = images[ids]
          caption[ids]['base'] = basename(images[ids])

        caption[ids]['ko'].append(c['korean'] )
        caption[ids]['en'].append(c['english'])

    except error as e:
      continue

    if not cls in classes:
      classes[cls] = dict ()
      classes[cls]['ids'] = list ()
      classes[cls]['num'] = 0

    classes[cls]['ids'].append (ids)

  for k in classes:
    random.shuffle (classes[k]['ids'])
    classes[k]['num'] = len(classes[k]['ids'])

  for k in caption:
    random.shuffle (caption[k]['ko'])
    random.shuffle (caption[k]['en'])

  json.dump(classes, open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_classes.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
  json.dump(caption, open(join(cfg.ROOT_DIR, cfg.DATA_DIR, 'full_caption.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

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

prepare ()
build ()

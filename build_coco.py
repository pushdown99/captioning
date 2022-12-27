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
cfg.ROOT_DIR     = 'data/COCO'
cfg.DATA_DIR     = 'annotations'
cfg.PREFIX       = '2014'
cfg.TRAIN_JSON   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'captions_train' + cfg.PREFIX + '.json')
cfg.VALID_JSON   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'captions_val'   + cfg.PREFIX + '.json')
cfg.IMAGE_JSON   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'images'         + cfg.PREFIX + '.json')
cfg.TRAIN_IMAGE  = join(cfg.ROOT_DIR, 'train2014') 
cfg.VALID_IMAGE  = join(cfg.ROOT_DIR, 'val2014') 
cfg.NORMAL_SIZE  = 1.0
cfg.FAT_SIZE     = 0.5
cfg.SMALL_SIZE   = 0.05
cfg.TRAIN_SIZE   = 0.8
cfg.VALID_SIZE   = 0.1
cfg.TEST_SIZE    = 0.1
cfg.CAPTION      = 5

def build_caption(f, d):
  fp = open(f, 'w')
  for k in d:
    if len(d[k]) < cfg.CAPTION:
      continue

    idx  = 0
    caption = list()
    for ko in d[k]:
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
      fp.write ('{}#{}\t{}\n'.format(k, i, c))

  fp.close()

def build_dataset(f, d):
  fp = open(f, 'w')
  for k in d:
    if len(d[k]) < cfg.CAPTION:
      continue
    fp.write ('{}\n'.format(k))
  fp.close()

#######################################

def time_start (t, f):
  print (t+':', f)
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed:', elapsed_time)

def prepare ():
  images = dict ()

  print (join(cfg.TRAIN_IMAGE, '*.json'))
  for p in tqdm(glob(join(cfg.TRAIN_IMAGE, '*.jpg'), recursive=True), desc='train' + cfg.PREFIX):
    #ids = int(basename(p).split('_')[2].split('.')[0])
    ids = basename(p)
    images [ids] = p
  for p in tqdm(glob(join(cfg.VALID_IMAGE, '*.jpg'), recursive=True), desc='valid' + cfg.PREFIX):
    #ids = int(basename(p).split('_')[2].split('.')[0])
    ids = basename(p)
    images [ids] = p

  print (len(images))
  json.dump(images, open(cfg.IMAGE_JSON,   'w', encoding='utf-8'), indent=2, ensure_ascii=False)

  trains = dict ()
  data = json.load(codecs.open(cfg.TRAIN_JSON, 'r', 'utf-8-sig'))
  for c in data['annotations']:
    if cfg.PREFIX == '2014':
      ids = 'COCO_train2014_{:012d}.jpg'.format(c['image_id'])
    else:
      ids = '{:012d}.jpg'.format(c['image_id'])

    caption = c['caption']

    if not ids in trains:
      trains[ids] = list()

    trains[ids].append(caption)

  print (len(trains))
  k = list(trains.keys())[0]
  print (trains[k])

  valids = dict ()
  data = json.load(codecs.open(cfg.VALID_JSON, 'r', 'utf-8-sig'))
  for c in data['annotations']:
    if cfg.PREFIX == '2014':
      ids = 'COCO_val2014_{:012d}.jpg'.format(c['image_id'])
    else:
      ids = '{:012d}.jpg'.format(c['image_id'])
    caption = c['caption']

    if not ids in valids:
      valids[ids] = list()

    valids[ids].append(caption)

  print (len(valids))

  n_total_full   = int(len(trains | valids) * cfg.NORMAL_SIZE)
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

  print (n_total_full, n_train_full, n_valid_full, n_test_full)
  print (n_total_fat, n_train_fat, n_valid_fat, n_test_fat)
  print (n_total_small, n_train_small, n_valid_small, n_test_small)

  data = trains | valids

  total_full   = dict()
  train_full   = dict()
  valid_full   = dict()
  test_full    = dict()

  total_fat    = dict()
  train_fat    = dict()
  valid_fat    = dict()
  test_fat     = dict()

  total_small  = dict()
  train_small  = dict()
  valid_small  = dict()
  test_small   = dict()


  for cnt, (k, v) in enumerate(data.items()):
    if cnt >= 0 and cnt < n_total_full:
      total_full[k] = v
    if cnt >= 0 and cnt < n_train_full:
      train_full[k] = v
    if cnt >= n_train_full and cnt < n_valid_full:
      valid_full[k] = v
    if cnt >= n_valid_full and cnt < n_test_full:
      test_full[k] = v

    if cnt >= 0 and cnt < n_total_fat:
      total_fat[k] = v
    if cnt >= 0 and cnt < n_train_fat:
      train_fat[k] = v
    if cnt >= n_train_fat    and cnt < n_valid_fat:
      valid_fat[k] = v
    if cnt >= n_valid_fat    and cnt < n_test_fat:
      test_fat[k] = v

    if cnt >= 0 and cnt < n_total_small:
      total_small[k] = v
    if cnt >= 0 and cnt < n_train_small:
      train_small[k] = v
    if cnt >= n_train_small  and cnt < n_valid_small:
      valid_small[k] = v
    if cnt >= n_valid_small  and cnt < n_test_small:
      test_small[k] = v

  print ('====')
  print ('full: ', len(train_full), len(valid_full), len(test_full))
  print ('fat:  ', len(train_fat), len(valid_fat), len(test_fat))
  print ('small:', len(train_small), len(valid_small), len(test_small))

  f_total_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption' + cfg.PREFIX + '_full.txt' )
  f_total_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption' + cfg.PREFIX + '_fat.txt'  )
  f_total_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'caption' + cfg.PREFIX + '_small.txt')

  f_train_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train' + cfg.PREFIX + '_full.txt'   )
  f_train_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train' + cfg.PREFIX + '_fat.txt'    )
  f_train_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'train' + cfg.PREFIX + '_small.txt'  )

  f_valid_full   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid' + cfg.PREFIX + '_full.txt'   )
  f_valid_fat    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid' + cfg.PREFIX + '_fat.txt'    )
  f_valid_small  = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'valid' + cfg.PREFIX + '_small.txt'  )

  f_test_full    = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test' + cfg.PREFIX + '_full.txt'    )
  f_test_fat     = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test' + cfg.PREFIX + '_fat.txt'     )
  f_test_small   = join(cfg.ROOT_DIR, cfg.DATA_DIR, 'test' + cfg.PREFIX + '_small.txt'   )

  build_caption (f_total_full,  total_full )
  build_dataset (f_train_full,  train_full )
  build_dataset (f_valid_full,  valid_full )
  build_dataset (f_test_full,   test_full  )

  build_caption (f_total_fat,   total_fat  )
  build_dataset (f_train_fat,   train_fat  )
  build_dataset (f_valid_fat,   valid_fat  )
  build_dataset (f_test_fat,    test_fat   )

  build_caption (f_total_small, total_small)
  build_dataset (f_train_small, train_small)
  build_dataset (f_valid_small, valid_small)
  build_dataset (f_test_small,  test_small )

prepare ()
#build ()

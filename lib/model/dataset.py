import tensorflow as tf

import os
import re
import gc
import json
import glob
import codecs
import csv
import random
import string
import numpy as np
import pandas as pd
import collections
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from os.path import join
from _pickle import dump,load

from dataclasses import dataclass
from config import cfg_dataset_size

@dataclass
class Box:
  class_index: int
  class_name: str
  corners: np.ndarray
  
  def __repr__(self):
    return "[class=%s (%f,%f,%f,%f)]" % (self.class_name, self.corners[0], self.corners[1], self.corners[2], self.corners[3])

  def __str__(self):
    return repr(self)

class Dataset:
  def __init__(self, cfg, verbose = True):
    self.verbose = verbose
    self.cfg     = cfg

    #os.makedirs('files', exist_ok=True)

    # Run-time environment
    cuda_available = tf.test.is_built_with_cuda()
    gpu_available  = tf.config.list_physical_devices('GPU')
    print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
    print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
    print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

  def df_load_descriptions(self, df, title='[title]', number_of_captions = 5):
    mapping = dict()
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc=title):
      image_id = r['image_id']
      caption  = r['caption']

      if image_id not in mapping:
        mapping[image_id] = list()

      if len(mapping[image_id]) >= number_of_captions:
        continue

      mapping[image_id].append(caption)

    #self.clean_descriptions (mapping) // english only

    return mapping

  def clean_descriptions(self, descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
      for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc = [word for word in desc if len(word)>1]
        desc = [word for word in desc if word.isalpha()]
        desc_list[i] =  ' '.join(desc)

  def to_vocabulary (self, descriptions):
    all_desc = set()
    for key in descriptions.keys():
      [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

class DatasetVoc:
  def __init__(self, split, dir = "datasets/VOCdevkit/VOC2007", feature_pixels = 16, augment = True, shuffle = True, allow_difficult = False, cache = True):

    self.split = split
    self.dir   = dir
    self.class_index_to_name = self.get_classes (self.split)
    self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
    self.num_classes = len(self.class_index_to_name)
    self.filepaths = self.get_filepaths (self.split)
    self.num_samples = len(self.filepaths)
    self.gt_boxes_by_filepath = self.get_ground_truth_boxes(filepaths = self.filepaths, allow_difficult = False) 
    self.i = 0
    self.iterable_filepaths = self.filepaths.copy()
    self.feature_pixelsa = feature_pixels
    self.augment = True
    self.augment = True
    self.shuffle = True
    self.cache   = True
    self.unaugmented_cached_sample_by_filepath = {}
    self.augmented_cached_sample_by_filepath = {}

  def __iter__(self):
    self.i = 0
    if self.shuffle:
      random.shuffle(self.iterable_filepaths)
    return self

  def __next__(self):
    if self.i >= len(self.iterable_filepaths):
      raise StopIteration

    filepath = self.iterable_filepaths[self.i]
    self.i += 1

    flip = random.randint(0, 1) != 0 if self.augment else 0
    cached_sample_by_filepath = self.augmented_cached_sample_by_filepath if flip else self.unaugmented_cached_sample_by_filepath
  
    if filepath in cached_sample_by_filepath:
      sample = cached_sample_by_filepath[filepath]
    else:
      sample = self.generate_training_sample(filepath = filepath, flip = flip)
    if self.cache:
      cached_sample_by_filepath[filepath] = sample

    return sample

  def generate_training_sample(self, filepath, flip):
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    scaled_gt_boxes = []
    for box in self.gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([
          box.corners[0],
          original_width - 1 - box.corners[3],
          box.corners[2],
          original_width - 1 - box.corners[1]
        ]) 
      else:
        corners = box.corners
      scaled_box = Box(
        class_index = box.class_index,
        class_name = box.class_name,
        corners = corners * scale_factor 
      )
      scaled_gt_boxes.append(scaled_box)

    anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = scaled_image_data.shape, feature_pixels = self.feature_pixels)
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

    return TrainingSample(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
      image_data = scaled_image_data,
      image = scaled_image,
      filepath = filepath
    )

  def get_classes (self, split):
    dir = join(self.dir, "ImageSets", "Main")
    classes = set([ os.path.basename(path).split("_")[0] for path in Path(dir).glob("*_" + split + ".txt") ])
    class_index_to_name = { (1 + v[0]): v[1] for v in enumerate(sorted(classes)) }
    class_index_to_name[0] = "background"
    return class_index_to_name

  def get_filepaths(self, split):
    dir = join(self.dir, "ImageSets", "Main", split + ".txt")
    with open(dir) as f:
      basenames = [ line.strip() for line in f.readlines() ] # strip newlines
    image_paths = [ join(self.dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
    return image_paths

  def get_ground_truth_boxes(self, filepaths, allow_difficult):
    gt_boxes_by_filepath = {}
    for filepath in filepaths:
      basename = os.path.splitext(os.path.basename(filepath))[0]
      annotation_file = join(self.dir, "Annotations", basename) + ".xml"
      tree = ET.parse(annotation_file)
      root = tree.getroot()
      size = root.find("size")
      depth = int(size.find("depth").text)
      boxes = []
      for obj in root.findall("object"):
        is_difficult = int(obj.find("difficult").text) != 0
        if is_difficult and not allow_difficult:
          continue  # ignore difficult examples unless asked to include them
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
        y_min = int(bndbox.find("ymin").text) - 1
        x_max = int(bndbox.find("xmax").text) - 1
        y_max = int(bndbox.find("ymax").text) - 1
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = self.class_name_to_index[class_name], class_name = class_name, corners = corners)
        boxes.append(box)
      gt_boxes_by_filepath[filepath] = boxes
    return gt_boxes_by_filepath 

  def get_data (self):
    self.get_categories('datasets/NIA/annotations/4-2/', False, 2)

  def get_categories (self, dir, force = False, indent = None):
    name  = self.cfg.DATA.NAME
    cfile = 'files/{}_categories.json'.format(name)

    if force or not os.path.isfile(cfile):
      categories = dict()
      for filename in tqdm(glob.glob(dir + '*.json')):
        object = json.load(codecs.open(filename, 'r', 'utf-8-sig'))

        for c in object['categories']:
          cid   = c['id']
          cname = c['name']
          if not cid in categories:
            categories[cid] = cname

      json.dump(categories, open(cfile, 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)

    else:
      categories = json.load(codecs.open(cfile, 'r', 'utf-8-sig'))

    print (categories)
    print (categories['2'])
    return categories

class flickr8k (Dataset):
  def __init__(self, config, verbose = True, force = True, indent = None):
    super().__init__ (config, verbose)

    name    = config['name']
    images  = config['images_dir']
    caption = config['caption_file']
    trains  = config['train_file']
    valids  = config['valid_file']
    tests   = config['test_file']
    limit1  = config['train_limit']
    limit2  = config['valid_limit']
    limit3  = config['test_limit']
    number_of_captions    = config['number_of_captions']

    self.get_data (name, images, caption, trains, valids, tests, limit1, limit2, limit3, force, number_of_captions, indent)

  def get_data (self, name, images, caption, trains, valids, tests, limit1, limit2, limit3, force = True, number_of_captions = 5, indent = None):
    if not force and os.path.isfile('files/{}_descr.json'.format(name)):
      return

    df  = pd.read_csv(caption, lineterminator='\n', names=['image_id', 'caption'], sep='\t')
    df1 = pd.read_csv(trains,  lineterminator='\n', names=['image_id'])
    df2 = pd.read_csv(valids,  lineterminator='\n', names=['image_id'])
    df3 = pd.read_csv(tests,   lineterminator='\n', names=['image_id'])
    df.loc[:,'image_id']  = images[0]+'/'+df.image_id.str.split('#').str[0]
    df1.loc[:,'image_id'] = images[0]+'/'+df1.image_id.str.split('#').str[0]
    df2.loc[:,'image_id'] = images[0]+'/'+df2.image_id.str.split('#').str[0]
    df3.loc[:,'image_id'] = images[0]+'/'+df3.image_id.str.split('#').str[0]
    df1 = df.loc[df['image_id'].isin(list(df1.image_id))]
    df2 = df.loc[df['image_id'].isin(list(df2.image_id))]
    df3 = df.loc[df['image_id'].isin(list(df3.image_id))]

    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds) 
    dx  = list(df['caption'])            # list for caption data
    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds1 = dict(list(ds1.items())[:limit1])
    ds2 = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    ds2 = dict(list(ds2.items())[:limit2])
    ds3 = self.df_load_descriptions (df3, 'Test dataset ', number_of_captions)
    ds3 = dict(list(ds3.items())[:limit3])

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    json.dump(ds, open('files/{}_descr.json'.format(name), 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds1,open('files/{}_train.json'.format(name), 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds2,open('files/{}_valid.json'.format(name), 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds3,open('files/{}_test.json'.format(name),  'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(dx, open('files/{}_text.json'.format(name),  'w', encoding='utf-8'), indent=indent, ensure_ascii=False)

class nia (Dataset):
  def __init__(self, cfg, size='full', verbose = True, force = True, indent = 2):
    super().__init__ (cfg, verbose)
    cfg_dataset_size (size)

    name    = cfg.DATA.NAME
    root    = cfg.DATA.ROOT
    data    = join(root, cfg.DATA.DATA)
    jsons   = join(root, cfg.DATA.JSON_DIR)
    images  = join(root, cfg.DATA.IMAGE_DIR)
    caption = join(data, cfg.DATA.CAP_FILE)
    trains  = join(data, cfg.DATA.TRAIN_FILE)
    valids  = join(data, cfg.DATA.VALID_FILE)
    tests   = join(data, cfg.DATA.TEST_FILE)
    number_of_captions = cfg.DATA.CAPTIONS

    self.get_data  (name, images, caption, trains, valids, tests, force, number_of_captions, indent)

  def get_data (self, name, images, caption, trains, valids, tests, force, number_of_captions, indent):
    df  = pd.read_csv(caption, lineterminator='\n', names=['image_id', 'caption'], sep='\t')
    df1 = pd.read_csv(trains,  lineterminator='\n', names=['image_id'])
    df2 = pd.read_csv(valids,  lineterminator='\n', names=['image_id'])
    df3 = pd.read_csv(tests,   lineterminator='\n', names=['image_id'])

    df.loc [:,'image_id'] = images+'/' +  df.image_id.str.split('#').str[0]
    df1.loc[:,'image_id'] = images+'/' + df1.image_id.str.split('#').str[0]
    df2.loc[:,'image_id'] = images+'/' + df2.image_id.str.split('#').str[0]
    df3.loc[:,'image_id'] = images+'/' + df3.image_id.str.split('#').str[0]

    df1 = df.loc[df['image_id'].isin(list(df1.image_id))]
    df2 = df.loc[df['image_id'].isin(list(df2.image_id))]
    df3 = df.loc[df['image_id'].isin(list(df3.image_id))]

    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds) 
    dx  = list(df['caption'])            # list for caption data

    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds2 = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    ds3 = self.df_load_descriptions (df3, 'Test dataset ', number_of_captions)

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    cfg = self.cfg
    json.dump(ds, open(cfg.DATA.DESC_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds1,open(cfg.DATA.TRAIN_JSON, 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds2,open(cfg.DATA.VALID_JSON, 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds3,open(cfg.DATA.TEST_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(dx, open(cfg.DATA.TEXT_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)

class coco (Dataset):

  def __init__(self, cfg, size='full', verbose = True, force = True, indent = None):
    super().__init__ (cfg, verbose)
    cfg_dataset_size (size)

    name    = cfg.DATA.NAME
    root    = cfg.DATA.ROOT
    data    = join(root, cfg.DATA.DATA)
    caption = join(data, cfg.DATA.CAP_FILE)
    trains  = join(data, cfg.DATA.TRAIN_FILE)
    valids  = join(data, cfg.DATA.VALID_FILE)
    tests   = join(data, cfg.DATA.TEST_FILE)
    number_of_captions = cfg.DATA.CAPTIONS

    self.get_data  (name, root, caption, trains, valids, tests, force, number_of_captions, indent)

  def get_data (self, name, root, caption, trains, valids, tests, force, number_of_captions, indent):

    df  = pd.read_csv(caption, lineterminator='\n', names=['image_id', 'caption'], sep='\t')
    df1 = pd.read_csv(trains,  lineterminator='\n', names=['image_id'])
    df2 = pd.read_csv(valids,  lineterminator='\n', names=['image_id'])
    df3 = pd.read_csv(tests,   lineterminator='\n', names=['image_id'])

    df.loc [:,'image_id'] = root + '/' +  df.image_id.str.split('_').str[1] + '/' +  df.image_id.str.split('#').str[0]
    df1.loc[:,'image_id'] = root + '/' + df1.image_id.str.split('_').str[1] + '/' + df1.image_id.str.split('#').str[0]
    df2.loc[:,'image_id'] = root + '/' + df2.image_id.str.split('_').str[1] + '/' + df2.image_id.str.split('#').str[0]
    df3.loc[:,'image_id'] = root + '/' + df3.image_id.str.split('_').str[1] + '/' + df3.image_id.str.split('#').str[0]

    df1 = df.loc[df['image_id'].isin(list(df1.image_id))]
    df2 = df.loc[df['image_id'].isin(list(df2.image_id))]
    df3 = df.loc[df['image_id'].isin(list(df3.image_id))]

    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds)
    dx  = list(df['caption'])            # list for caption data

    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds2 = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    ds3 = self.df_load_descriptions (df3, 'Test dataset ', number_of_captions)

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    cfg = self.cfg
    json.dump(ds, open(cfg.DATA.DESC_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds1,open(cfg.DATA.TRAIN_JSON, 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds2,open(cfg.DATA.VALID_JSON, 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(ds3,open(cfg.DATA.TEST_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)
    json.dump(dx, open(cfg.DATA.TEXT_JSON , 'w', encoding='utf-8'), indent=indent, ensure_ascii=False)



class voc (Dataset):

  def __init__(self, config, verbose = True, force = True, indent = None):
    super().__init__ (config, verbose)

    train_data = DatasetVoc(dir = config['dataset_dir'], split = 'trainval', augment = True,  shuffle = True,  cache = True )
    eval_data  = DatasetVoc(dir = config['dataset_dir'], split = 'test',     augment = False, shuffle = False, cache = False)
    dump(train_data, open('files/{}_rpn_train.pkl'.format(config['name']), 'wb'))
    dump(eval_data,  open('files/{}_rpn_eval.pkl'.format(config['name']),  'wb'))


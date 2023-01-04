from __future__ import  absolute_import

import lib
import os, re, sys, time, math, json, random, codecs, argparse
import statistics
from pprint import pprint
from colorama import init, deinit, Back, Fore
from tqdm import tqdm
from glob import glob
from pickle import dump,load
from os.path import join, basename
from json.decoder import JSONDecodeError as error
from easydict import EasyDict as edict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.

class cfg:
  data       = 'nia'
  root_dir   = 'data/NIA'
  data_dir   = join(root_dir, 'annotations')
  image_dir  = join(root_dir, 'images')
  objects    = join(root_dir, 'objects.json')
  json42     = join(root_dir, 'download/4-2')
  json43     = join(root_dir, 'download/4-3')
  json44     = join(root_dir, 'download/4-4')

  min_per_class = 50
  num_per_class = 200
  limits = 10000
  split  = [0.8, 0.1]

  trainvaltest  = list ()
  _trainvaltest = join(root_dir, 'trainvaltest.txt')
  _trainval  = join(root_dir, 'trainval.txt')
  _train     = join(root_dir, 'train.txt')
  _val       = join(root_dir, 'val.txt')
  _test      = join(root_dir, 'test.txt')

  ######################################################

  object_id   = dict ()
  id_object   = dict ()
  instances   = dict ()
  images      = dict ()
  files       = list ()
  data42      = dict ()
  data43      = dict ()
  data44      = dict ()
  cls_files   = dict ()
  obj_files   = dict ()
  cls_stats   = dict ()
  obj_stats   = dict ()
  pred_dicts  = dict ()
  obj_dicts   = dict ()

  _object_id  = join(root_dir, 'object_id.json')
  _id_object  = join(root_dir, 'id_object.json')
  _instances  = join(root_dir, 'instances.json')
  _images     = join(root_dir, 'images.json')
  _files      = join(root_dir, 'files.txt')
  _json42     = join(root_dir, '4-2.json')
  _json43     = join(root_dir, '4-3.json')
  _json44     = join(root_dir, '4-4.json')
  _cls_files  = join(root_dir, 'classes.file.json')
  _obj_files  = join(root_dir, 'objects.file.json')
  _cls_stats  = join(root_dir, 'classes.stats.json')
  _obj_stats  = join(root_dir, 'objects.stats.json')
  _pred_dicts = join(root_dir, 'predicates.dict.json')
  _obj_dicts  = join(root_dir, 'objects.dict.json')


  def _state_dict(self):
    return {k: getattr(self, k) for k, _ in cfg.__dict__.items() if not k.startswith('_')}

  def _parse(self, kwargs):
    state_dict = self._state_dict()
    for k, v in kwargs.items():
      if k not in state_dict:
        raise ValueError('unknown option: "--%s"' % k)
      setattr(self, k, v)

    print('------ configuration ------')
    pprint(self._state_dict())
    print('---------- end ------------')

opt = cfg ()

#########################################################################################

def get_images (dir_):
  return {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(dir_, '**/*.jpg'), recursive=True), desc='images   ')}

def get_objects ():
  if not os.path.exists(cfg.objects):
    print ('object file not exist:', cfg.objects)
    sys.exit()

  id_object = json.load(codecs.open(cfg.objects, 'r', 'utf-8-sig'))
  object_id = { object: str(id) for (id, object) in id_object.items() }

  return id_object, object_id

def load_json (f):
  return json.load(codecs.open(f, 'r', 'utf-8-sig'))

def get_jsons (dir_, desc):
  return {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(dir_, '**/*.json'), recursive=True), desc=desc)}

def put_json (d, f):
  json.dump(d, open(f, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def put_text (d, f):
  fp = open(f, 'w')
  fp.write('\n'.join(d))
  fp.close()

def get_instances ():
  opt.images = get_images (opt.image_dir)
  opt.id_object, opt.object_id = get_objects ()
  opt.data42 = get_jsons (opt.json42, '4-2 jsons')
  opt.data43 = get_jsons (opt.json43, '4-3 jsons')
  opt.data44 = get_jsons (opt.json44, '4-4 jsons')

  for i, k in enumerate(tqdm(opt.data42, desc='instances')):
    if not k in opt.images:
      continue
    if not k in opt.data43:
      continue
    if not k in opt.data44:
      continue

    p42 = opt.data42[k]
    if os.path.getsize(p42) <= 0:
      continue
    d42 = json.load(codecs.open(p42, 'r', 'utf-8-sig'))
    if len(d42['annotations']) == 0:
      continue


    p43 = opt.data43[k]
    if os.path.getsize(p43) <= 0:
      continue
    d43 = json.load(codecs.open(p43, 'r', 'utf-8-sig'))
    if len(d43['annotations']) == 0:
      continue

    p44 = opt.data44[k]
    if os.path.getsize(p44) <= 0:
      continue
    d44 = json.load(codecs.open(p44, 'r', 'utf-8-sig'))
    if len(d44['annotations']) == 0:
      continue

    if not k in opt.instances:
      opt.instances[k] = dict()
      opt.instances[k]['bbox']   = list()
      opt.instances[k]['ko']     = list()
      opt.instances[k]['en']     = list()
      opt.instances[k]['matrix'] = d44['annotations'][0]['matrix']

    opt.instances[k]['width']  = d42['images'][0]['width']
    opt.instances[k]['height'] = d42['images'][0]['height']
    opt.instances[k]['image']  = opt.images[k]
    opt.instances[k]['p42']    = opt.data42[k]
    opt.instances[k]['p43']    = opt.data43[k]
    opt.instances[k]['p44']    = opt.data44[k]

    name = basename(opt.images[k]).split('.')[0]
    cls  = name.split('(')[0].split('_')[2]
    ins  = name.split('(')[1].split(')')[0]

    if not cls in opt.cls_files:
      opt.cls_files[cls] = list()
    opt.cls_files[cls].append(name)

    for d in d42['annotations']:
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

      if not str(d['category_id']) in opt.id_object:
        print ('category_id (', d['category_id'], ') not found! ', p)
        continue

      obj = opt.id_object[str(d['category_id'])]
      an['name'] = obj

      if not obj in opt.obj_files:
        opt.obj_files[obj] = list()
      opt.obj_files[obj].append(name)

      opt.instances[k]['bbox'].append(an)

    for d in d43['annotations'][0]['text']:
      opt.instances[k]['ko'].append(d['korean'])
      opt.instances[k]['en'].append(d['english'])

    for d in d44['annotations'][0]['matrix']:
      src  = '<empty>' if type(d['source']) != str     or len(d['source']) <= 0     else d['source'].lower().strip()
      dest = '<empty>' if type(d['target']) != str     or len(d['target']) <= 0     else d['target'].lower().strip()
      pred = '<empty>' if type(d['m_relation']) != str or len(d['m_relation']) <= 0 else d['m_relation'].lower().strip()

      if not pred in opt.pred_dicts:
        opt.pred_dicts [pred] = 0
      opt.pred_dicts [pred] += 1

      if not src in opt.obj_dicts:
        opt.obj_dicts [src] = 0
      opt.obj_dicts [src] += 1

      if not dest in opt.obj_dicts:
        opt.obj_dicts [dest] = 0
      opt.obj_dicts [dest] += 1

#########################################################################################

def prepare (**kwargs):
  opt._parse(kwargs)

  _t = time.process_time()

  get_instances()
  opt.files = [basename(v['image']).split('.')[0] for k, v in opt.instances.items()]

  opt.obj_files = { k:list(set(v)) for k, v in opt.obj_files.items()}

  opt.cls_stats = { k: len(opt.cls_files[k]) for k in opt.cls_files}
  opt.cls_stats = dict(sorted(opt.cls_stats.items(), key=lambda item: item[1], reverse=True))

  opt.obj_stats = { k: len(opt.obj_files[k]) for k in opt.obj_files}
  opt.obj_stats = dict(sorted(opt.obj_stats.items(), key=lambda item: item[1], reverse=True))

  opt.pred_dicts = dict(sorted(opt.pred_dicts.items(), key=lambda item: item[1], reverse=True))
  opt.obj_dicts  = dict(sorted(opt.obj_dicts.items(), key=lambda item: item[1], reverse=True))

  put_json (opt.images, opt._images)
  put_json (opt.files, opt._files)
  put_json (opt.instances, opt._instances)
  put_json (opt.object_id, opt._object_id)
  put_json (opt.id_object, opt._id_object)
  put_json (opt.cls_files, opt._cls_files)
  put_json (opt.obj_files, opt._obj_files)
  put_json (opt.cls_stats, opt._cls_stats)
  put_json (opt.obj_stats, opt._obj_stats)
  put_json (opt.pred_dicts, opt._pred_dicts)
  put_json (opt.obj_dicts, opt._obj_dicts)

  put_text (opt.files, opt._files)

  print ('---------------------------')
  print ('images    :', len(opt.images))
  print ('files     :', len(opt.files))
  print ('instances :', len(opt.instances))
  print ('object_id :', len(opt.object_id))
  print ('id_object :', len(opt.id_object))
  print ('classes   :', len(opt.cls_files))
  print ('objects   :', len(opt.obj_files))
  print ('cls_stats :', len(opt.cls_stats))
  print ('obj_stats :', len(opt.obj_stats))
  print ('pred_dicts:', len(opt.pred_dicts))
  print ('obj_dicts :', len(opt.obj_dicts))
  print ('---------------------------')

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed:', _elapsed)
  print ('')

def build_nia_dataset ():
  opt.cls_files = load_json (opt._cls_files)

  opt.trainvaltest = [[ l for l in v[:opt.num_per_class] if len(v) >= opt.min_per_class] for k, v in opt.cls_files.items()]
  opt.trainvaltest = [x for sublist in opt.trainvaltest for x in sublist]
  random.shuffle(opt.trainvaltest)

  opt.trainvaltest = opt.trainvaltest[:opt.limits]

  total = len(opt.trainvaltest)
  size1 = int(total * opt.split[0])
  size2 = int(total * (opt.split[0]+opt.split[1]))

  opt.train = opt.trainvaltest[:size1]
  opt.val = opt.trainvaltest[size1:size2]
  opt.test = opt.trainvaltest[size2:]
  opt.trainval = opt.train + opt.val

  put_text (opt.trainvaltest, opt._trainvaltest)
  put_text (opt.trainval, opt._trainval)
  put_text (opt.test, opt._test)

  print ('trainvaltest:', len(opt.trainvaltest))
  print ('train       :', len(opt.train))
  print ('val         :', len(opt.val))
  print ('trainval    :', len(opt.trainval))
  print ('test        :', len(opt.test))

def build (**kwargs):
  opt._parse(kwargs)

  if opt.data == 'nia':
    build_nia_dataset()

if __name__ == '__main__':
  import fire

  fire.Fire()

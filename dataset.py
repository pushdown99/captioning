from __future__ import  absolute_import, annotations

import lib
import re
import numpy
import string
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
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
from collections import defaultdict
from importlib import reload
from nltk import flatten

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.

class cfg:
  data       = 'nia'
  root_dir   = 'dataset/NIA'
  data_dir   = join(root_dir, 'annotations')
  image_dir  = join(root_dir, 'images')
  objects    = join(root_dir, 'objects.json')
  json42     = join(root_dir, 'download/4-2')
  json43     = join(root_dir, 'download/4-3')
  json44     = join(root_dir, 'download/4-4')

  min_per_class = 50
  num_per_class = 3000
  limits = 30000
  split  = [0.8, 0.1]

  trainvaltest    = list ()

  n_caption = 10
  lang = 'ko' # ko en
  thuru = 3                            # 3 | 4

  _trainvaltest   = join(root_dir, 'trainvaltest.txt')
  _trainval       = join(root_dir, 'trainval.txt')
  _train          = join(root_dir, 'train.txt')
  _val            = join(root_dir, 'val.txt')
  _test           = join(root_dir, 'test.txt')

  __trainvaltest  = join(root_dir, 'trainvaltest.json')
  __trainval      = join(root_dir, 'trainval.json')
  __train         = join(root_dir, 'train.json')
  __val           = join(root_dir, 'val.json')
  __test          = join(root_dir, 'test.json')

  _c_trainvaltest = join(root_dir, 'c_trainvaltest.json')
  _c_trainval     = join(root_dir, 'c_trainval.json')
  _c_train        = join(root_dir, 'c_train.json')
  _c_val          = join(root_dir, 'c_val.json')
  _c_test         = join(root_dir, 'c_test.json')
  _c_text         = join(root_dir, 'c_text.json')

  ######################################################

  object_id   = dict ()
  id_object   = dict ()
  instances   = dict ()
  images      = dict ()
  captions    = dict ()
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
  categories  = dict ()
  inv_weight  = dict ()
  dicts       = dict ()
  word_dict = dict ()
  unalias     = dict ()

  _object_id  = join(root_dir, 'object_id.json')
  _id_object  = join(root_dir, 'id_object.json')
  _instances  = join(root_dir, 'instances.json')
  _images     = join(root_dir, 'images.json')
  _captions   = join(root_dir, 'captions.json')
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
  _categories = join(root_dir, 'categories.json')
  _inv_weight = join(root_dir, 'inverse_weight.json')
  _dicts      = join(root_dir, 'dict.json')
  _word_dict = join(root_dir, 'word_dict.json')
  _kor2eng      = join(root_dir, 'kor2eng.json')
  _eng2kor    = join(root_dir, 'eng2kor.json')
  _unalias    = join(root_dir, 'unalias.json')
  _anno       = join(root_dir, 'anno.json')
  _replace    = join(root_dir, 'replace.json')


  def _clean_up (self):
    delattr(cfg, 'data42')
    delattr(cfg, 'data43')
    delattr(cfg, 'data44')
    delattr(cfg, 'json42')
    delattr(cfg, 'json43')
    delattr(cfg, 'json44')
    delattr(cfg, '_json42')
    delattr(cfg, '_json43')
    delattr(cfg, '_json44')

  def _state_dict (self):
    return {k: getattr(self, k) for k, _ in cfg.__dict__.items() if not k.startswith('_')}

  def _parse (self, kwargs):
    state_dict = self._state_dict()
    for k, v in kwargs.items():
      if k not in state_dict:
        raise ValueError('unknown option: "--%s"' % k)
      setattr(self, k, v)

  def _set (self, k, v):
    setattr(self, k, v)

  def _print (self):
    print('----------- configuration -----------')
    pprint(self._state_dict())
    print('--------------- end -----------------')

opt = cfg ()

#########################################################################################

def update ():
  opt._trainvaltest = join(opt.root_dir, 'trainvaltest.txt')
  opt._trainval   = join(opt.root_dir, 'trainval.txt')
  opt._train      = join(opt.root_dir, 'train.txt')
  opt._val        = join(opt.root_dir, 'val.txt')
  opt._test       = join(opt.root_dir, 'test.txt')

  opt.__trainvaltest = join(opt.root_dir, 'trainvaltest.json')
  opt.__trainval   = join(opt.root_dir, 'trainval.json')
  opt.__train      = join(opt.root_dir, 'train.json')
  opt.__val        = join(opt.root_dir, 'val.json')
  opt.__test       = join(opt.root_dir, 'test.json')

  opt._c_trainvaltest = join(opt.root_dir, 'c_trainvaltest.json')
  opt._c_trainval     = join(opt.root_dir, 'c_trainval.json')
  opt._c_train        = join(opt.root_dir, 'c_train.json')
  opt._c_val          = join(opt.root_dir, 'c_val.json')
  opt._c_test         = join(opt.root_dir, 'c_test.json')
  opt._c_text         = join(opt.root_dir, 'c_text.json')

  opt._object_id  = join(opt.root_dir, 'object_id.json')
  opt._id_object  = join(opt.root_dir, 'id_object.json')
  opt._instances  = join(opt.root_dir, 'instances.json')
  opt._images     = join(opt.root_dir, 'images.json')
  opt._captions   = join(opt.root_dir, 'captions.json')
  opt._files      = join(opt.root_dir, 'files.txt')
  opt._json42     = join(opt.root_dir, '4-2.json')
  opt._json43     = join(opt.root_dir, '4-3.json')
  opt._json44     = join(opt.root_dir, '4-4.json')
  opt._cls_files  = join(opt.root_dir, 'classes.file.json')
  opt._obj_files  = join(opt.root_dir, 'objects.file.json')
  opt._cls_stats  = join(opt.root_dir, 'classes.stats.json')
  opt._obj_stats  = join(opt.root_dir, 'objects.stats.json')
  opt._pred_dicts = join(opt.root_dir, 'predicates.dict.json')
  opt._obj_dicts  = join(opt.root_dir, 'objects.dict.json')
  opt._categories = join(opt.root_dir, 'categories.json')
  opt._dicts      = join(opt.root_dir, 'dict.json')
  opt._kor2eng    = join(opt.root_dir, 'kor2eng.json')
  opt._eng2kor    = join(opt.root_dir, 'eng2kor.json')
  opt._trans2     = join(opt.root_dir, 'trans2.json')
  opt._unalias    = join(opt.root_dir, 'unalias.json')
  opt._anno       = join(opt.root_dir, 'anno.json')

def clean_descriptions (descriptions):
  table = str.maketrans('', '', string.punctuation)
  for i, v in enumerate (descriptions):
    desc = descriptions[i]
    desc = desc.split()
    desc = [word.lower() for word in desc]
    desc = [w.translate(table) for w in desc]
    desc = [word for word in desc if len(word)>1]
    desc = [word for word in desc if word.isalpha()]
    descriptions[i] =  ' '.join(desc)
  return descriptions

def get_images (dir_):
  return {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(dir_, '**/*.jpg'), recursive=True), desc='images   ')}

def get_objects ():
  if not os.path.exists(opt.objects):
    print ('object file not exist:', opt.objects)
    sys.exit()

  id_object = json.load(codecs.open(opt.objects, 'r', 'utf-8-sig'))
  object_id = { object: str(id) for (id, object) in id_object.items() }

  return id_object, object_id

def load_json (f):
  return json.load(codecs.open(f, 'r', 'utf-8-sig'))

def get_jsons (dir_, desc):
  return {basename(p).split('.')[0].split('_')[0]+'_'+basename(p).split('.')[0].split('_')[1]:p for p in tqdm(glob(join(dir_, '**/*.json'), recursive=True), desc=desc)}

if sys.version_info.major == 3 and 4 <= sys.version_info.minor <= 6:
  import _make_iterencode
  json.encoder._make_iterencode = _make_iterencode._make_iterencode
  indent = (2, None)

def put_json (d, f):
  json.dump(d, open(f, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def put_text (d, f):
  fp = open(f, 'w')
  fp.write('\n'.join(d))
  fp.close()

#########################################################################################

forbidden = [
  '이 사진의 주제는',
  '표지에',
  '제목의',
  '제목이'
]
def is_malformed_sentence (s):
  for f in forbidden:
    if s.find (f)>= 0:
      return True
  return False

replaces = load_json (opt._replace)
kor2eng  = load_json (opt._kor2eng)
eng2kor  = load_json (opt._eng2kor)

def word_refine (ko):
  if ko in kor2eng:
    en = kor2eng[ko]
    if en in eng2kor:
      return eng2kor[en]
  return ko

def word_split (ko, offset):
  if offset !=  0:
    w1 = ko [:-offset]
    w2 = ko [-offset:]
  else:
    w1 = ko
    w2 = ''

  return w1, w2

def word_spacing (ko, offset):
  w1, w2 = word_split (ko, offset)

  if w1 in kor2eng:
    return word_refine (w1) + ' ' + word_refine (w2)
  elif w2 in kor2eng:
    return word_refine (w1) + ' ' + word_refine (w2)

  return ko

def sentence_refine (sentence, flag='pre', lang='ko'):
  for r in replaces[flag]:
    sentence = sentence.replace(r['src'], r['dst'])

  if lang == 'ko':
    ps = sentence.split()
    pl = len (ps)

    for i, p in enumerate (ps):
      if not p in kor2eng:
        length = len (p)
        for offset in range (0, length):
          ps[i] = word_spacing (p, offset)
          if ps[i].find (' ') >= 0:
            break
      else:
        ps[i] = word_refine (p)
    sentence = ' '.join(ps)
  else:
    ps = sentence.split()

    table = str.maketrans('', '', string.punctuation)
    desc = [word.lower() for word in ps]
    desc = [w.translate(table) for w in desc]
    desc = [word for word in desc if len(word)>1]
    desc = [word for word in desc if word.isalpha()]
    sentence = ' '.join(desc)
    
  return sentence

def merge_bbox (box1, box2):
  x1 = min(box1[0], box2[0])
  y1 = min(box1[1], box2[1])
  x2 = max(box1[2], box2[2])
  y2 = max(box1[3], box2[3])

  return [x1, y1, x2, y2]

def convert_box_wh_xy (b):
  b = list(map(int, b))
  return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def convert_box_xy_wh (b):
  b = list(map(int, b))
  return [b[0], b[1], b[2]-b[0], b[3]+b[1]]

def convert_box_wh_yx (b):
  b = list(map(int, b))
  return [b[1], b[0], b[1]+b[3], b[0]+b[2]]

ENTITY_LENGTH = 14

def relation_verify (r, o, c, i, offset = True):
  if isinstance(r[o], str) and len(r[o]) < ENTITY_LENGTH:
    d = r[o]
    if d in kor2eng and kor2eng[d] in eng2kor and kor2eng[d] in c:
      l = numpy.where(i == kor2eng[d])[0]
      if offset:
        return kor2eng[d], int(l[0])
      else:
        return kor2eng[d], int(l[len(l)-1])
      
  return r[o], -1 

  
def class_to_relationships (objects, relations):
  classes = [o['class'] for o in objects]
  indexes = numpy.array(classes)
  data = list ()
  _l  = list ()
  _ul = list ()

  for i, r in enumerate (relations):
    sub_id, sub_idx = relation_verify (r, 'entity1', classes, indexes)
    obj_id, obj_idx = relation_verify (r, 'entity2', classes, indexes, False)
    predicate = r['relation'].lower()
    _ul = list ()

    if sub_idx >= 0 and obj_idx >= 0:
      d = {'sub_id': sub_idx, 'sub': sub_id, 'predicate': predicate, 'obj_id': obj_idx, 'obj': obj_id}
      data.append (d)
      _l.append (i)
    if sub_idx < 0: _ul.append (sub_id)
    if obj_idx < 0: _ul.append (obj_id)

  return data, _l, _ul

def get_nia_instances ():
  opt.images = get_images (opt.image_dir)
  opt.id_object, opt.object_id = get_objects ()
  opt.data42 = get_jsons (opt.json42, '4-2 jsons')
  opt.data43 = get_jsons (opt.json43, '4-3 jsons')
  opt.data44 = get_jsons (opt.json44, '4-4 jsons')

  opt.obj_dicts = {k:0 for k in opt.object_id.keys()}

  anno       = dict ()
  trans2     = dict ()

  opt.dicts['idx2word'] = list ()

  for i, k in enumerate(tqdm(opt.data42, desc='instances')):
    if not k in opt.images: continue
    if not k in opt.data43: continue
    if not k in opt.data44: continue

    p42 = opt.data42[k]
    p43 = opt.data43[k]
    p44 = opt.data44[k]

    if os.path.getsize(p42) <= 0: continue
    if os.path.getsize(p43) <= 0: continue
    if os.path.getsize(p44) <= 0: continue

    d42 = json.load(codecs.open(p42, 'r', 'utf-8-sig'))
    if len(d42['annotations']) == 0: continue

    d43 = json.load(codecs.open(p43, 'r', 'utf-8-sig'))
    if len(d43['annotations']) == 0: continue

    d44 = json.load(codecs.open(p44, 'r', 'utf-8-sig'))
    if len(d44['annotations']) == 0: continue

    if not k in opt.instances:
      opt.instances[k] = dict()
      opt.instances[k]['height'] = d42['images'][0]['height']
      opt.instances[k]['width']  = d42['images'][0]['width']
      opt.instances[k]['path']   = basename(opt.images[k])

      opt.instances[k]['image']  = opt.images[k]
      opt.instances[k]['json2']  = opt.data42[k]
      opt.instances[k]['json3']  = opt.data43[k]
      opt.instances[k]['json4']  = opt.data44[k]

      opt.instances[k]['relationships'] = list ()
      opt.instances[k]['regions']       = list ()
      opt.instances[k]['objects']       = list ()
      opt.instances[k]['captions']      = dict ()
      #opt.instances[k]['42'] = [{'box': list(map(int, o['bbox'])), 'id': o['category_id'], 'class': opt.id_object[str(o['category_id'])]} for o in d42['annotations']]
      opt.instances[k]['42'] = [{'box': o['bbox'], 'id': o['category_id'], 'class': opt.id_object[str(o['category_id'])]} for o in d42['annotations']]
      opt.instances[k]['43'] = d43['annotations'][0]['text']
      opt.instances[k]['44'] = d44['annotations'][0]['text']

    if sum ([ 1 if o['korean']=='' else 0 for o in opt.instances[k]['43']]) > 0:
      del opt.instances[k]
      continue
    if sum ([ 1 if o['english']=='' else 0 for o in opt.instances[k]['43']]) > 0:
      del opt.instances[k]
      continue

    name = basename(opt.images[k]).split('.')[0]
    cls  = name.split('(')[0].split('_')[2]
    ins  = name.split('(')[1].split(')')[0]

    ## objects
    #opt.instances[k]['objects'] = [{'box':[o['box'][0], o['box'][1], o['box'][0]+o['box'][2], o['box'][1]+o['box'][3]], 'class': o['class']} for o in opt.instances[k]['42']]
    opt.instances[k]['objects'] = [{'box': convert_box_wh_xy (o['box']), 'class': o['class']} for o in opt.instances[k]['42']]
    opt.instances[k]['relationships'], _l, _ul  = class_to_relationships (opt.instances[k]['42'], opt.instances[k]['44'])

    for d in opt.instances[k]['relationships']:
      pred = d['predicate']
      sub  = d['sub']
      obj  = d['obj']
      if not pred in opt.pred_dicts:
        opt.pred_dicts [pred] = 0
      opt.pred_dicts [pred] += 1

      #if not sub in opt.obj_dicts:
      #  opt.obj_dicts [sub] = 0
      opt.obj_dicts [sub] += 1

      #if not obj in opt.obj_dicts:
      #  opt.obj_dicts [obj] = 0
      opt.obj_dicts [obj] += 1

    for u in _ul:
      if not u in opt.unalias:
        opt.unalias[u] = 0
      opt.unalias[u] += 1

    ## captions
    ## relations & regions
    for i, _i in enumerate (_l):
      ko = opt.instances[k]['44'][_i]['korean']
      ko = sentence_refine (ko,  'pre', 'ko')
      ko = sentence_refine (ko, 'post', 'ko')

      en = opt.instances[k]['44'][_i]['english']
      en = sentence_refine (en,  'pre', 'en')
      en = sentence_refine (en, 'post', 'en')

      region = dict ()
      sub  = opt.instances[k]['relationships'][i]['sub_id']
      obj  = opt.instances[k]['relationships'][i]['obj_id']
      pred = opt.instances[k]['relationships'][i]['predicate']
      box1 = opt.instances[k]['objects'][sub]['box']
      box2 = opt.instances[k]['objects'][obj]['box']
      region['box'] = merge_bbox (box1, box2)

      if opt.lang == 'ko':
        region['phrase']  = ko.split ()
        for p in region['phrase']: 
          opt.dicts['idx2word'].append (p)
          if not p in opt.word_dict: opt.word_dict[p] = 0
          opt.word_dict[p] += 1
        region['phrase2'] = en.split ()
      else:
        region['phrase']  = en.split ()
        for p in region['phrase']: 
          opt.dicts['idx2word'].append (p)
          if not p in opt.word_dict: opt.word_dict[p] = 0
          opt.word_dict[p] += 1
        region['phrase2'] = ko.split ()

      opt.instances[k]['regions'].append (region)

    ###########################################################
    #
    # normal condition.
    #
    ko43 = [ c['korean']  for c in opt.instances[k]['43']]
    en43 = [ c['english'] for c in opt.instances[k]['43']]
    if is_malformed_sentence (' '.join(ko43)):
      del opt.instances[k]
      continue

    ko44 = [ c['korean']  for c in opt.instances[k]['44']]
    en44 = [ c['english'] for c in opt.instances[k]['44']]
    if is_malformed_sentence (' '.join(ko44)):
      del opt.instances[k]
      continue

    if len(ko43) != opt.n_caption or len(en43) != opt.n_caption or len(ko44) != opt.n_caption or len(en44) != opt.n_caption:
      del opt.instances[k]
      continue

    if not cls in opt.cls_files:
      opt.cls_files[cls] = list()
    opt.cls_files[cls].append(name)

    for obj in [ o['class'] for o in opt.instances[k]['42']]:
      if not obj in opt.obj_files:
        opt.obj_files[obj] = list()
      opt.obj_files[obj].append(name)

    ########################################################

    if opt.thuru == 3:
      korean  = ko43
      english = en43
    else:
      korean  = ko44
      english = en44

    opt.instances[k]['captions']['korean'] = list ()
    for ko in korean:
      ko = sentence_refine (ko,  'pre', 'ko')
      ko = sentence_refine (ko, 'post', 'ko')
      opt.instances[k]['captions']['korean'].append (ko)

    opt.instances[k]['captions']['english'] = list ()
    for en in english:
      en = sentence_refine (en,  'pre', 'en')
      en = sentence_refine (en, 'post', 'en')
      opt.instances[k]['captions']['english'].append (en)

    if opt.lang == 'ko':
      opt.captions[k] = opt.instances[k]['captions']['korean']
    else:
      opt.captions[k] = opt.instances[k]['captions']['english']

  opt.dicts['idx2word'] = list(set(opt.dicts['idx2word']))
  opt.dicts['word2idx'] = { k:i for i, k in enumerate (opt.dicts['idx2word']) }

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))
  print ('')
  print('dicts', len(opt.dicts['idx2word']))

  opt.unalias = dict(sorted(opt.unalias.items(), key=lambda item: item[1], reverse=True))

'''
    for d in d42['annotations']:
      an = dict() # [ x, y, maxx, maxy]
      an['box'] = [0, 0, 0, 0]
      an['box'][0] = int(d['bbox'][0])
      an['box'][1] = int(d['bbox'][1])
      an['box'][2] = int(d['bbox'][0]+d['bbox'][2])
      an['box'][3] = int(d['bbox'][1]+d['bbox'][3])

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
      ko  = sentence_refine (d['korean'],  'pre', 'ko')
      ko  = sentence_refine (ko, 'post', 'ko')

      en  = sentence_refine (d['english'], 'pre', 'en')
      en  = sentence_refine (en, 'post', 'en')

      #d['korean']  = d['korean'].replace('. .','.')
      #d['korean']  = d['korean'].replace('.','')
      #d['english'] = d['english'].replace('. .','.')
      #d['english'] = d['english'].replace('.','')
      #opt.instances[k]['ko'].append(d['korean'])
      #opt.instances[k]['en'].append(d['english'])
      opt.instances[k]['ko'].append(ko)
      opt.instances[k]['en'].append(en)

    if len(opt.instances[k]['ko']) != opt.n_caption:
      del opt.instances[k]
      continue

    if is_malformed_sentence (' '.join(opt.instances[k]['ko'])):
      del opt.instances[k]
      continue

    if not cls in opt.cls_files:
      opt.cls_files[cls] = list()
    opt.cls_files[cls].append(name)

    if opt.lang == 'ko':
      #opt.captions[k] = list(map(str.lower, opt.instances[k]['ko']))
      #opt.captions[k] = clean_descriptions (opt.instances[k]['ko'])
      kor = list ()
      for ko in list(map(str.lower, opt.instances[k]['ko'])):
        ko  = sentence_refine (ko)
        ko  = sentence_refine (ko, 'post')
        kor.append (ko)
      opt.captions[k] = kor
    else:
      #opt.captions[k] = list(map(str.lower, opt.instances[k]['en']))
      opt.captions[k] = clean_descriptions (opt.instances[k]['en'])

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
'''

def get_coco_instances ():
  trains = {basename(p).split('.')[0]:p for p in tqdm(glob(join(opt.image_dir[0], '*.jpg'), recursive=True), desc='train2017')}
  vals   = {basename(p).split('.')[0]:p for p in tqdm(glob(join(opt.image_dir[1], '*.jpg'), recursive=True), desc='val2017  ')}
  tests  = {basename(p).split('.')[0]:p for p in tqdm(glob(join(opt.image_dir[2], '*.jpg'), recursive=True), desc='test2017 ')}
  opt.images = trains | vals | tests
#  opt.images = trains | vals

  opt.id_object, opt.object_id = get_objects ()

  obj1 = json.load(codecs.open(join(opt.data_dir, 'captions_train2017.json'), 'r', 'utf-8-sig'))
  obj2 = json.load(codecs.open(join(opt.data_dir, 'captions_val2017.json'), 'r', 'utf-8-sig'))
  obj = obj1['annotations'] + obj2['annotations']

  for d in tqdm(obj, desc='captions '):
    k = '{:012d}'.format(d['image_id'])
    if not k in opt.captions:
      opt.captions[k] = list()

    cap = d['caption'].strip().lower()
    cap = cap.replace('.','')
    cap = cap.replace('\n','')
    opt.captions[k].append(cap)

    opt.captions[k] = clean_descriptions (opt.captions[k])

  obj1 = load_json (opt.train_inst)
  obj2 = load_json (opt.val_inst)

  # duplicated
  obj = obj1['annotations'] + obj2['annotations']
  img = obj1['images'] + obj2['images']

  #data = {'{:012d}'.format(o['image_id']):o for o in obj}
  path = {'{:012d}'.format(d['id']):d for d in img}

  #for k, v in tqdm(data.items(), desc='instance '):
  for v in tqdm(obj, desc='instance '):
    k = '{:012d}'.format(v['image_id'])
    if not k in opt.images:
      continue

    if not k in opt.instances:
      opt.instances[k] = dict ()
      opt.instances[k]['bbox'] = list()
      opt.instances[k]['en'] = opt.captions[k]

    opt.instances[k]['width']  = path[k]['width']
    opt.instances[k]['height'] = path[k]['height']
    opt.instances[k]['image']  = path[k]['file_name']

    box  = list(map(int, v['bbox'])) # x y w h

    inst = dict()
    bbox = [0 for i in range(4)]   # xmin ymin width height
    bbox[0] = int(box[1])          # ymin
    bbox[1] = int(box[0])          # xmin
    bbox[2] = int(box[1]+box[3])   # ymax = miny + height
    bbox[3] = int(box[0]+box[2])   # xmax = minx + width
    inst['bbox']  = bbox # ymin xmin ymax xmax
    inst['xywh']  = box
    inst['name']  = opt.id_object[str(v['category_id'])]

    if not inst['name'] in opt.cls_files:
      opt.cls_files[inst['name']] = list()
      opt.obj_files[inst['name']] = list()

    opt.cls_files[inst['name']].append(k)
    opt.obj_files[inst['name']].append(k)

    opt.instances[k]['bbox'].append(inst)

def get_vg_instance ():
  opt.images = {basename(p).split('.')[0]:p for p in tqdm(glob(join(opt.image_dir, '*.jpg'), recursive=True), desc='images')}
  trains = json.load(codecs.open(join(opt.data_dir, 'train.json'), 'r', 'utf-8-sig'))
  tests  = json.load(codecs.open(join(opt.data_dir, 'test.json'), 'r', 'utf-8-sig'))
  opt.id_object, opt.object_id = get_objects () # 150

  instances = dict ()
  for d in tqdm(trains+tests, desc='trains'):
    k = basename (d['path']).split('.')[0]
    if not k in opt.images:
      continue

    if not k in opt.instances:
      opt.instances[k] = dict ()
      opt.instances[k]['bbox'] = list()

    if not 'person' in opt.cls_files:
      opt.cls_files['person'] = list()
    opt.cls_files['person'].append(k)


    opt.instances[k]['width']  = d['width']
    opt.instances[k]['height'] = d['height']
    opt.instances[k]['image']  = d['path']

    for i, o in enumerate (d['objects']):
      box = list(map(int, o['box'])) # x y mx my
      
      inst = dict()
      bbox = [0 for i in range(4)]   # xmin ymin width height
      bbox[0] = int(box[1])          # ymin
      bbox[1] = int(box[0])          # xmin
      bbox[2] = int(box[3])          # ymax = miny + height
      bbox[3] = int(box[2])          # xmax = minx + width
      inst['bbox']  = bbox # ymin xmin ymax xmax
      inst['xywh']  = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
      inst['name']  = o['class']

      opt.instances[k]['bbox'].append (inst)


#########################################################################################

def prepare_print ():
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

def prepare_dataset ():
  opt.files = [basename(v['image']).split('.')[0] for k, v in opt.instances.items()]

  opt.obj_files = { k:list(set(v)) for k, v in opt.obj_files.items()}

  opt.cls_stats = { k: len(opt.cls_files[k]) for k in opt.cls_files}
  opt.cls_stats = dict(sorted(opt.cls_stats.items(), key=lambda item: item[1], reverse=True))

  opt.obj_stats = { k: len(opt.obj_files[k]) for k in opt.obj_files}
  opt.obj_stats = dict(sorted(opt.obj_stats.items(), key=lambda item: item[1], reverse=True))

def prepare_voc_dataset ():
  print('')

def prepare_coco_dataset ():
  get_coco_instances()
  prepare_dataset ()

  put_json (opt.images, opt._images)
  put_json (opt.files, opt._files)
  put_json (opt.instances, opt._instances)
  put_json (opt.object_id, opt._object_id)
  put_json (opt.id_object, opt._id_object)
  put_json (opt.cls_files, opt._cls_files)
  put_json (opt.obj_files, opt._obj_files)
  put_json (opt.cls_stats, opt._cls_stats)
  put_json (opt.obj_stats, opt._obj_stats)
  put_json (opt.captions, opt._captions)
  put_text (opt.files, opt._files)

  prepare_print ()

def prepare_vg_dataset ():
#  obj = json.load(codecs.open(join(opt.data_dir, 'categories.json'), 'r', 'utf-8-sig'))
#  objects = { i:o for i, o in enumerate (obj['object']) }
#  json.dump(objects, open(join(opt.root_dir, 'objects.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
#  prediates = { i:o for i, o in enumerate (obj['predicate']) }
#  json.dump(prediates, open(join(opt.root_dir, 'prediates.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
   
  get_vg_instance ()
  prepare_dataset ()

  put_json (opt.images, opt._images)
  put_json (opt.instances, opt._instances)
  put_json (opt.object_id, opt._object_id)
  put_json (opt.id_object, opt._id_object)
  put_json (opt.cls_files, opt._cls_files)

#  obj = json.load(codecs.open(join(opt.data_dir, 'train.json'), 'r', 'utf-8-sig'))
#  for idx, d in enumerate (obj): 
#    cls = dict()
#    for i, o in enumerate (d['objects']):
#      #l.append(o['class'])
#      #print (ii, r['box'], r['class'])
#      cls[str(i)] = o['class']
#    #cls = tuple (l)
#
#    for i, r in enumerate (d['relationships']):
#      print(idx, cls[str(r['sub_id'])], r['predicate'], cls[str(r['obj_id'])], r['sub_id'], r['obj_id'])
      
# [{'sub_id': 1, 'predicate': 'wear', 'obj_id': 0},
#    print (d['relationships'])
  #json.dump(d['object'], open(join(opt.root_dir, 'objects.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

WEIGHT = 100.0

def invert_weight (dicts):
  weights = [1.0 / 1 if w == 0 else w for w in list(dicts.values())]
  weights = [w / sum(weights) * WEIGHT for w in weights]

  return { k: weights[i] for i, k in enumerate (dicts)}

def prepare_nia_dataset (**kwargs):
  get_nia_instances()

  print ('captions',  len(opt.captions))
  print ('instances', len(opt.instances))
  prepare_dataset ()

  opt.pred_dicts = dict(sorted(opt.pred_dicts.items(), key=lambda item: item[1], reverse=True))
  opt.obj_dicts  = dict(sorted(opt.obj_dicts.items(), key=lambda item: item[1], reverse=True))
  opt.word_dict  = dict(sorted(opt.word_dict.items(), key=lambda item: item[1], reverse=True))

  opt.categories['predicate'] = list(opt.pred_dicts.keys())
  opt.categories['object']    = list(opt.object_id.keys())

  opt.inv_weight['predicate'] = invert_weight (opt.pred_dicts)
  opt.inv_weight['object']    = invert_weight (opt.obj_dicts)

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
  put_json (opt.captions, opt._captions)
  put_json (opt.categories, opt._categories)
  put_json (opt.unalias, opt._unalias)
  put_json (opt.word_dict, opt._word_dict)
  put_json (opt.inv_weight, opt._inv_weight)
  put_text (opt.files, opt._files)

  prepare_print ()

#########################################################################################

def prepare (**kwargs):
  opt._parse(kwargs)
  callback ('prepare', opt.data)

#########################################################################################

def build_caption (l, f, to_text=False):
  opt.images    = load_json (opt._images)
  opt.captions  = load_json (opt._captions)

  if opt.data == 'nia':
    data = {opt.images[basename(k).split('_')[0]+'_'+basename(k).split('_')[1]]:opt.captions[basename(k).split('_')[0]+'_'+basename(k).split('_')[1]][:opt.n_caption] for k in l}
  else:
    data = {opt.images[k]:opt.captions[k][:opt.n_caption] for k in l}

  if to_text:
    d = list([v for v in data.values()])
    [x for sublist in d for x in sublist]
    put_json (flatten([v for v in data.values()]), opt._c_text)

  put_json (data, f)
    
def build_categories ():
  obj       = list(load_json (opt._object_id).keys())
  pred      = list(load_json (opt._pred_dicts).keys())
  instances = load_json (opt._instances)

  categories = dict ()
  dicts      = dict ()
  unalias    = dict ()
  anno       = dict ()
  trans2     = dict ()

  dicts['idx2word'] = list ()

  categories['predicate'] = pred
  categories['object']    = obj

  put_json (categories, opt._categories)

  _t = time.process_time()

  cnt = 0
  for id, key in tqdm(enumerate (instances)):
    mats = instances[key]['matrix']
    text = instances[key]['text']
    bbox = instances[key]['bbox']
    classes = [ b['name'] for b in instances[key]['bbox']]
    indexes = numpy.array(classes)

    for i, m in enumerate (mats):
      source   = ''
      target   = ''
      relation = ''

      if isinstance(m['source'], str) and len(m['source'])<14:
        d = m['source'].replace(' ','')
        if d in kor2eng:
          #dicts['idx2word'].append (kor2eng[d])
          source = kor2eng[d]
        else:
          if not d in unalias:
            unalias[d] = 0
          unalias[d] += 1
      else:
        d = m['source']
        if not d in unalias:
          unalias[d] = 0
        unalias[d] += 1
        #print (d, key)

      if isinstance(m['target'], str) and len(m['source'])<14:
        d = m['target'].replace(' ','')
        if d in kor2eng:
          #dicts['idx2word'].append (kor2eng[d])
          target = kor2eng[d]
        else:
          if not d in unalias:
            unalias[d] = 0
          unalias[d] += 1
      else:
        d = m['target']
        if not d in unalias:
          unalias[d] = 0
        unalias[d] += 1
        #print (d, key)

      if isinstance(m['m_relation'], str):
        relation = m['m_relation'].lower()

      if len(source) > 0  and len(target) > 0 and len(relation) > 0:
        # normal case
        sidx = -1
        tidx = -1
        sloc = list ()
        tloc = list ()

        if source in classes:
          sloc = numpy.where(indexes == source)[0]
          sidx = int(sloc[0])
          #sidx = classes.index(source)

        if target in classes:
          tloc = numpy.where(indexes == target)[0]
          tidx = int(tloc[len(tloc)-1])
          #tidx = classes.index(target)

        if sidx >= 0 and tidx >= 0:

          if not key in anno :
            anno[key] = dict ()
            anno[key]['height'] = instances[key]['height']
            anno[key]['width']  = instances[key]['width']
            anno[key]['path']   = basename(instances[key]['image'])
            anno[key]['id']     = int(id + 1)
            anno[key]['relationships'] = list ()
            anno[key]['original']      = list ()
            anno[key]['regions']       = list ()
            anno[key]['objects']       = list ()

          relationships = dict ()
          relationships['sub_id'] = sidx
          relationships['predicate'] = relation
          relationships['obj_id'] = tidx
          anno[key]['relationships'].append(relationships)
          anno[key]['original'].append('{}_{}_{}'.format(source, relation, target))

          regions = dict ()
          regions['box'] = merge_bbox (bbox[sidx]['xywh'], bbox[tidx]['xywh'])

          korean  = sentence_refine (text[i]['korean'])
          korean  = sentence_refine (korean, 'post')
          english = text[i]['english']

          regions['phrase']  = korean.split()
          for phrase in regions['phrase']:
            dicts['idx2word'].append (phrase)

          regions['phrase2'] = english.split()

          trans2[key] = dict ()
          trans2[key]['source'] = text[i]['korean']
          trans2[key]['target'] = ' '.join(regions['phrase'])

          anno[key]['regions'].append(regions)

    if key in anno:
      for b in bbox:
        objects = dict ()
        objects['box'] = [ b['xywh'][0], b['xywh'][1], b['xywh'][0]+b['xywh'][2], b['xywh'][1]+b['xywh'][3]]
        objects['class'] = b['name']
        anno[key]['objects'].append (objects)

  dicts['idx2word'] = list(set(dicts['idx2word']))
  dicts['word2idx'] = { k:i for i, k in enumerate (dicts['idx2word']) }

  _elapsed = time.process_time() - _t
  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))
  print ('')
  print('dicts', len(dicts['idx2word']))

  unalias = dict(sorted(unalias.items(), key=lambda item: item[1], reverse=True))

  ######################################################################################

  put_json (dicts, opt._dicts)
  put_json (unalias, opt._unalias)
  put_json (anno, opt._anno)
  put_json (trans2, opt._trans2)

  print ('anno:', len(anno))
  return anno

def build_dataset ():
  opt.cls_files = load_json (opt._cls_files)
  opt.instances = load_json (opt._instances)

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
  put_text (opt.train, opt._train)
  put_text (opt.val, opt._val)
  put_text (opt.test, opt._test)

  if opt.data == 'nia' or opt.data == 'coco':
    build_caption (opt.trainvaltest, opt._c_trainvaltest, True)
    build_caption (opt.trainval, opt._c_trainval)
    build_caption (opt.train, opt._c_train)
    build_caption (opt.val, opt._c_val)
    build_caption (opt.test, opt._c_test)


  if opt.data == 'nia':
    trainvaltest = list ()
    trainval     = list ()
    train        = list ()
    val          = list ()
    test         = list ()

    trainvaltest = list([opt.instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in opt.trainvaltest])
    trainval = list([opt.instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in opt.trainval])
    train = list([opt.instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in opt.train])
    val = list([opt.instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in opt.val])
    test = list([opt.instances[k.split('_')[0]+'_'+k.split('_')[1]] for k in opt.test])
    
#  #  #anno = build_categories ()
#
#    for v in opt.trainvaltest:
#      k = v.split('_')[0]+'_'+v.split('_')[1]
#      if not k in anno:
#        continue
#      trainvaltest.append (anno[k])
#  
#    for v in opt.train:
#      k = v.split('_')[0]+'_'+v.split('_')[1]
#      if not k in anno:
#        continue
#      train.append (anno[k])
#  
#    for v in opt.trainval:
#      k = v.split('_')[0]+'_'+v.split('_')[1]
#      if not k in anno:
#        continue
#      trainval.append (anno[k])
#  
#    for v in opt.val:
#      k = v.split('_')[0]+'_'+v.split('_')[1]
#      if not k in anno:
#        continue
#      val.append (anno[k])
#  
#    for v in opt.test:
#      k = v.split('_')[0]+'_'+v.split('_')[1]
#      if not k in anno:
#        continue
#      test.append (anno[k])
  
    put_json (trainvaltest, opt.__trainvaltest)
    put_json (train, opt.__train)
    put_json (trainval, opt.__trainval)
    put_json (val, opt.__val)
    put_json (test, opt.__test)

  ##############################################

  print ('trainvaltest(txt) :', len(opt.trainvaltest))
  print ('train       (txt) :', len(opt.train))
  print ('val         (txt) :', len(opt.val))
  print ('trainval    (txt) :', len(opt.trainval))
  print ('test        (txt) :', len(opt.test))

  if opt.data == 'nia':
    print ('trainvaltest(json):', len(trainvaltest))
    print ('train       (json):', len(train))
    print ('val         (json):', len(val))
    print ('trainval    (json):', len(trainval))
    print ('test        (json):', len(test))


def build_voc_dataset ():
  print('')

def build_coco_dataset ():
  build_dataset ()

def build_vg_dataset ():
  build_dataset ()

def build_nia_dataset ():
  build_dataset ()

def build (**kwargs):
  opt._parse(kwargs)
  callback ('build', opt.data)

def callback (op, data):
  cb = [
    ['build',   'nia',  build_nia_dataset],
    ['build',   'voc',  build_voc_dataset],
    ['build',   'coco', build_coco_dataset],
    ['build',   'vg',   build_vg_dataset],

    ['prepare', 'nia',  prepare_nia_dataset],
    ['prepare', 'voc',  prepare_voc_dataset],
    ['prepare', 'coco', prepare_coco_dataset],
    ['prepare', 'vg',   prepare_vg_dataset],
  ]

  if data != 'nia':
    opt._clean_up ()

  if data == 'coco':
    opt.root_dir = 'dataset/COCO'
    opt.data_dir = join(opt.root_dir, 'annotations')
    opt.image_dir = [join(opt.root_dir, 'train2017'), join(opt.root_dir, 'val2017'), join(opt.root_dir, 'test2017')]
    opt.objects = join(opt.root_dir, 'objects.json')
    opt.train_inst = join(opt.data_dir, 'instances_train2017.json')
    opt.val_inst = join(opt.data_dir, 'instances_val2017.json')

  elif data == 'voc':
    opt.root_dir = 'dataset/VOC/VOC2007'
    opt.data_dir = join(opt.root_dir, 'annotations')
    opt.image_dir = join(opt.root_dir, 'images')

  elif data == 'vg':
    opt.root_dir = 'dataset/GENOME'
    opt.data_dir = join(opt.root_dir, 'top_150_50')
    opt.image_dir = join(opt.root_dir, 'VG_100K')
    opt.objects = join(opt.root_dir, 'objects.json')

  update ()
  opt._print ()

  for d in cb:
    if d[0] != op:
      continue
    if d[1] != data:
      continue

    d[2]()

if __name__ == '__main__':
  _t = time.process_time()

  import fire
  fire.Fire()

  _elapsed = time.process_time() - _t

  print ('')
  print ('elapsed: {:.2f} sec'.format(_elapsed))

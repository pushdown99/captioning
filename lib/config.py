import os
import json
import codecs
from os.path import join
from easydict import EasyDict as edict

cfg = edict()

cfg.DATA = edict()

cfg.DATA.NAME       = 'nia'
cfg.DATA.ROOT       = 'data/NIA'
cfg.DATA.DATA       = 'annotations'
cfg.DATA.DATA_SIZE  = 'full' # small | fat | full
cfg.DATA.JSON_DIR   = 'annotations/4-3'
cfg.DATA.IMAGE_DIR  = 'images'
cfg.DATA.DATA_DIR   = 'data'
cfg.DATA.MODEL_DIR  = 'model'
cfg.DATA.TEMP_DIR   = 'temp'
cfg.OBJECTS_JSON    = join(cfg.DATA.DATA_DIR, 'objects.json')
cfg.DATA.OBJECTS    = json.load(codecs.open(cfg.OBJECTS_JSON, 'r', 'utf-8-sig'))

def cfg_dataset_size (size):
  cfg.DATA.CAP_FILE   = 'caption_'     + size + '.txt'
  cfg.DATA.TRAIN_FILE = 'train_'       + size + '.txt'
  cfg.DATA.VALID_FILE = 'valid_'       + size + '.txt'
  cfg.DATA.TEST_FILE  = 'test_'        + size + '.txt'
  cfg.DATA.CAPTIONS   = 10

  cfg.DATA.DESC_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'description_' + size + '.json')
  cfg.DATA.TRAIN_JSON = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'train_'       + size + '.json')
  cfg.DATA.VALID_JSON = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'valid_'       + size + '.json')
  cfg.DATA.TEST_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'test_'        + size + '.json')
  cfg.DATA.TEXT_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'text_'        + size + '.json')
  cfg.DATA.TOKENIZE   = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'tokenize_'    + size + '.pkl')

cfg_dataset_size (cfg.DATA.DATA_SIZE)

cfg.MODEL = edict()
cfg.MODEL.NAME           = 'efficientnetb0'
cfg.MODEL.IMAGE_SHAPE    = (299,299)
cfg.MODEL.MAX_VOCAB_SIZE = 2000000
cfg.MODEL.SEQ_LENGTH     = 25
cfg.MODEL.BATCH_SIZE     = 64
cfg.MODEL.SHUFFLE_DIM    = 512
cfg.MODEL.EMBED_DIM      = 512
cfg.MODEL.FF_DIM         = 1024
cfg.MODEL.NUM_HEADS      = 6


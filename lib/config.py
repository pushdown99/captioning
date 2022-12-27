import os
from os.path import join
from easydict import EasyDict as edict

cfg = edict()

cfg.DATA = edict()
cfg.DATA.NAME       = 'coco'
cfg.DATA.ROOT       = 'data/COCO'
cfg.DATA.DATA       = 'annotations'
cfg.DATA.DATA_SIZE  = 'full' # small | fat | full
cfg.DARA_MODEL_DIR  = 'model'
cfg.DATA.PREFIX     = '2014'
cfg.DATA.CAPTIONS   = 5

def cfg_dataset_size (size):
  cfg.DATA.CAP_FILE   = 'caption' + cfg.DATA.PREFIX + '_' + size + '.txt'
  cfg.DATA.TRAIN_FILE = 'train'   + cfg.DATA.PREFIX + '_' + size + '.txt'
  cfg.DATA.VALID_FILE = 'valid'   + cfg.DATA.PREFIX + '_' + size + '.txt'
  cfg.DATA.TEST_FILE  = 'test'    + cfg.DATA.PREFIX + '_' + size + '.txt'

  cfg.DATA.IMAGE_JSON = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'images'       + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.DESC_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'description_' + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.TRAIN_JSON = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'train_'       + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.VALID_JSON = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'valid_'       + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.TEST_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'test_'        + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.TEXT_JSON  = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'text_'        + cfg.DATA.PREFIX + '_' + size + '.json')
  cfg.DATA.TOKENIZE   = join(cfg.DATA.ROOT, cfg.DATA.DATA, 'tokenize_'    + cfg.DATA.PREFIX + '_' + size + '.pkl')

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


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
from hanspell import spell_checker

cfg = edict ()
cfg.ROOT = 'data/NIA'
cfg.DATA = 'annotations'
cfg.CORPUS_JSON = join(cfg.ROOT, cfg.DATA, 'corpus.json')

def time_start ():
  return time.process_time()

def time_stop (t):
  elapsed_time = time.process_time() - t
  print ('Elapsed:', elapsed_time)

txt = '고양이1은 고양이2보다 오른쪽 아래에 위치해 있습니다.'
corpus = json.load(codecs.open(cfg.CORPUS_JSON,  'r', 'utf-8-sig'))

print (txt)

for k, v in corpus.items():  
  txt = txt.replace(k, v)

print (txt)

t = time_start ()
#sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
sent = '김철수는극중두인격의사나이이광수역을맡았다.철수는한국유일의태권도전승자를가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)

time_stop(t)

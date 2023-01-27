#!/usr/bin/python

import json
import codecs
from os.path import join, basename, exists

threshold = float(0.1)

blacklist_json = 'blacklist.json'
scores_json    = 'scores.json'

if exists (blacklist_json):
  blacklist = json.load(codecs.open(blacklist_json, 'r', 'utf-8-sig'))
else: blacklist = dict ()

print ('len', len(blacklist))

def get_id_from_path (f):
  f = basename(f)
  return f.split('_')[0]+'_'+f.split('_')[1]

data = json.load(codecs.open(scores_json, 'r', 'utf-8-sig'))
_cnt = 0
_sum = float(0)

for i, d in enumerate (data): # path, bleu1, predicted, actual
  id = get_id_from_path (d['path'])

  if d['bleu1'] <= threshold:
    print ('{} {:.2f} {}'.format(id, d['bleu1'], d['predicted']))
    blacklist[id] = float('{:.2f}'.format(d['bleu1']))
  else:
    _cnt += 1
    _sum += d['bleu1']
    print ('{} {:.2f}'.format(id, d['bleu1']))

print (i, _cnt, _sum, _sum/_cnt)

if _sum/_cnt > float(68):
  json.dump(blacklist, open(blacklist_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)



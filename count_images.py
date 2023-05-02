from glob import glob
from os.path import basename, join, relpath, realpath, abspath, exists
import json


def get_name_cls_ins (path):
  name = basename(path).split('.')[0]
  cls  = name.split('(')[0].split('_')[2:]
  cls = ' '.join(cls)
  ins  = name.split('(')[1].split(')')[0]

  return name, cls, ins

########################################################################

stats = dict()
count = 0

with open ('dataset/result.json', 'rb') as fp: result = json.load (fp)

for k, v in result['stats'].items():
  if not k in stats:
    stats [k] = dict()
    stats [k]['images'] = v['total']
    stats [k]['test'] = 0

for f in glob ('./images/*.jpg'):
  name, cls, ins = get_name_cls_ins (f)

  count += 1
  print (cls)
  stats [cls]['test'] += 1

print ('Test images:', count)
print ('Test images / categories:')
print (json.dumps(stats, indent=2, ensure_ascii=False))


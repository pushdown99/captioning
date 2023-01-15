import json
import codecs
from os.path import join
from tqdm import tqdm

obj1 = json.load(codecs.open(join('dataset/COCO/annotations','captions_train2017.json'), 'r', 'utf-8-sig'))
obj2 = json.load(codecs.open(join('dataset/COCO/annotations','captions_val2017.json'), 'r', 'utf-8-sig'))
obj = obj1['annotations'] + obj2['annotations']

captions = dict()
for d in tqdm(obj):
  k = '{:012d}'.format(d['image_id'])
  if not k in captions:
    captions[k] = list()

  cap = d['caption'].strip()
  cap = cap.replace('.','')
  cap = cap.replace('\n','')
  captions[k].append(cap)

print (len(captions))
key = list(captions.keys())[0]
print (key, captions[key])




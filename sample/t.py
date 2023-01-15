from msgspec.json import decode
from msgspec import Struct

d = 'data/COCO/annotations/instances_train2017.json'

  
data = decode(open(d, "rb"), type=dict{"info": Info, "licenses": [Licenses], "images": [Images], "annotations": [Annotations], "categories": [Categories])}))


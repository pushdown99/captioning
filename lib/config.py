import os
from os.path import join
from easydict import EasyDict as edict

cfg = edict()

cfg.DATA = edict()
cfg.DATA.NAME       = 'nia'
cfg.DATA.ROOT_DIR   = 'data/NIA'
cfg.DATA.DATA_DIR   = 'annotations'
cfg.DATA.DATA_SIZE  = 'full' # small | fat | full
cfg.DATA.JSON_DIR   = 'annotations/4-3'
cfg.DATA.IMAGE_DIR  = 'images'
cfg.DATA.CAP_FILE   = 'caption_'     + cfg.DATA.DATA_SIZE + '.txt'
cfg.DATA.TRAIN_FILE = 'train_'       + cfg.DATA.DATA_SIZE + '.txt'
cfg.DATA.VALID_FILE = 'valid_'       + cfg.DATA.DATA_SIZE + '.txt'
cfg.DATA.TEST_FILE  = 'test_'        + cfg.DATA.DATA_SIZE + '.txt'
cfg.DATA.CAPTIONS   = 10

cfg.DATA.DESC_JSON  = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'description_' + cfg.DATA.DATA_SIZE + '.json')
cfg.DATA.TRAIN_JSON = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'train_'       + cfg.DATA.DATA_SIZE + '.json')
cfg.DATA.VALID_JSON = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'valid_'       + cfg.DATA.DATA_SIZE + '.json')
cfg.DATA.TEST_JSON  = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'test_'        + cfg.DATA.DATA_SIZE + '.json')
cfg.DATA.TEXT_JSON  = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'text_'        + cfg.DATA.DATA_SIZE + '.json')
cfg.DATA.TOKENIZE   = join(cfg.DATA.ROOT_DIR, cfg.DATA.DATA_DIR, 'tokenize_'    + cfg.DATA.DATA_SIZE + '.pkl')

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

cfg.DATA.OBJECTS    = {
  "1"  : "person",
  "2"  : "bicycle",
  "3"  : "car",
  "4"  : "motorcycle",
  "5"  : "scooter",
  "6"  : "bus",
  "7"  : "truck",
  "8"  : "traffic light",
  "9"  : "fire hydrant",
  "10" : "fire extinguisher",
  "11" : "sign",
  "12" : "trash bin",
  "13" : "bench",
  "14" : "roof",
  "15" : "bird",
  "16" : "cat",
  "17" : "dog",
  "18" : "chicken",
  "19" : "backpack",
  "20" : "umbrella",
  "21" : "handbag",
  "22" : "tie",
  "23" : "suitcase",
  "24" : "muffler",
  "25" : "hat",
  "26" : "ball",
  "27" : "poles",
  "28" : "plate(skis)",
  "29" : "board",
  "30" : "drone",
  "31" : "pilates equipment",
  "32" : "treadmill",
  "33" : "dumbbell",
  "34" : "golf club",
  "35" : "billiards cue",
  "36" : "skating shoes",
  "37" : "tennis racket",
  "38" : "badminton racket",
  "39" : "moonwalker",
  "40" : "basketball hoop",
  "41" : "carabiner",
  "42" : "table tennis racket",
  "43" : "rice cooker",
  "44" : "gas stove",
  "45" : "pot",
  "46" : "pan",
  "47" : "microwave",
  "48" : "toaster",
  "49" : "knives",
  "50" : "chopping boards",
  "51" : "ladle",
  "52" : "silicon spatula",
  "53" : "rice spatula",
  "54" : "vegetable peeler",
  "55" : "box grater",
  "56" : "scissors",
  "57" : "bowl",
  "58" : "cutlery",
  "59" : "plate",
  "60" : "side dish",
  "61" : "tray",
  "62" : "mug",
  "63" : "refrigerator",
  "64" : "dish washer",
  "65" : "espresso machine",
  "66" : "purifier",
  "67" : "banana",
  "68" : "apple",
  "69" : "grape",
  "70" : "pear",
  "71" : "melon",
  "72" : "cucumber",
  "73" : "watermelon",
  "74" : "orange",
  "75" : "peach",
  "76" : "strawberry",
  "77" : "plum",
  "78" : "persimmon",
  "79" : "lettuce",
  "80" : "cabbage",
  "81" : "radish",
  "82" : "perilla leaf",
  "83" : "garlic",
  "84" : "onion",
  "85" : "spring onion",
  "86" : "carrot",
  "87" : "corn",
  "88" : "potato",
  "89" : "sweet potato",
  "90" : "egg plant",
  "91" : "tomato",
  "92" : "pumpkin",
  "93" : "squash",
  "94" : "chili",
  "95" : "pimento",
  "96" : "sandwich",
  "97" : "hamburger",
  "98" : "hotdog",
  "99" : "pizza",
  "100": "donut",
  "101": "cake",
  "102": "white bread",
  "103": "icecream",
  "104": "ttoke",
  "105": "tteokbokki",
  "106": "kimchi",
  "107": "gimbap",
  "108": "sushi",
  "109": "mandu",
  "110": "gonggibap",
  "111": "couch",
  "112": "mirror",
  "113": "window",
  "114": "table",
  "115": "lamp",
  "116": "door",
  "117": "chair",
  "118": "bed",
  "119": "toilet bowl",
  "120": "washstand",
  "121": "book",
  "122": "clock",
  "123": "doll",
  "124": "hair drier",
  "125": "toothbrush",
  "126": "hair brush",
  "127": "tv",
  "128": "laptop",
  "129": "mouse",
  "130": "keyboard",
  "131": "cell phone",
  "132": "watch",
  "133": "camera",
  "134": "speaker",
  "135": "fan",
  "136": "air conditioner",
  "137": "piano",
  "138": "tambourine",
  "139": "castanets",
  "140": "guitar",
  "141": "violin",
  "142": "flute",
  "143": "recorder",
  "144": "xylophone",
  "145": "ocarina",
  "146": "thermometer",
  "147": "sphygmomanometer",
  "148": "blood glucose meter",
  "149": "defibrillator",
  "150": "massage gun",

  "151": "ceiling",
  "152": "floor",
  "153": "wall",
  "156": "road",
  "160": "building",

  "154": "pillar",
  "155": "unknown2",
  "157": "unknown3",
  "158": "tree",
  "159": "unknown5",
  "161": "unknown6",

  "162": "shuttlecock",
  "163": "hula hoop",
  "164": "gripper",
  "165": "whisk",
  "166": "tongs",
  "167": "jujube",
  "168": "chestnut"
}


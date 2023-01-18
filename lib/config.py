import tensorflow as tf
from pprint import pprint
from os.path import join


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --data-dir='./data/'

num_gpus    = len(tf.config.list_physical_devices('GPU'))
num_workers = num_gpus * 4

class Config:
    data = 'nia'
    data_dir = 'dataset'
    captions = join(data_dir, 'captions.json')
    trainval = join(data_dir, 'c_trainval.json')
    train    = join(data_dir, 'c_train.json')
    val      = join(data_dir, 'c_val.json')
    test     = join(data_dir, 'c_test.json')
    text     = join(data_dir, 'c_text.json')
    tokenize = join(data_dir, 'tokenize.pkl')
    trained = ''
    sample= ''

    model = 'efficientnetb0'
    epoch = 1
    n_caption = 10 #

    num_workers      = num_workers
    test_num_workers = num_workers

    IMAGE_SHAPE    = (299,299)
    MAX_VOCAB_SIZE = 2000000
    SEQ_LENGTH     = 25
    BATCH_SIZE     = 64
    SHUFFLE_DIM    = 512
    EMBED_DIM      = 512
    FF_DIM         = 1024
    NUM_HEADS      = 6

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()

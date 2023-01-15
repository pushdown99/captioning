# image captioning

keras/Tensorflow Image Captioning application using CNN and Transformer as encoder/decoder.
In particulary, the architecture consists of three models:

A CNN: used to extract the image features. In this application, it used EfficientNetB0 pre-trained on imagenet.
A TransformerEncoder: the extracted image features are then passed to a Transformer encoder that generates a new representation of the inputs.
A TransformerDecoder: it takes the encoder output and the text data sequence as inputs and tries to learn to generate the caption.

(reference code: https://github.com/Dantekk/Image-Captioning

## Progress
- [x] Guide for Project Setup
- [x] Guide for Model Evaluation with pretrained model
- [x] Guide for Model Training
- [x] Uploading pretrained model and format-compatible datasets.

## Dataset
The model has been trained on train/val NIA dataset. You can download the dataset here. Note that test images are not required for this code to work.

Original dataset has (A) images and (B) validation images; for each image there is a number of captions between 1 and 10. I have preprocessing the dataset per to keep only images that have exactly 10 captions. In fact, the model has been trained to ensure that 10 captions are assigned for each image. After this filtering, the final dataset has (A) train images and (B) validation images.
Finally, I serialized the dataset into two json files which you can find in:

NIA_dataset/c_train.json
NIA_dataset/c_val.json

Each element in the c_train.json file has such a structure :
"NIA_dataset/images/IMG_0061865_(...).jpg": ["caption1", "caption2", "caption3", "caption4", ..."caption10"], ...

In same way in the c_val.json :
"NIA_dataset/images/IMG_0061865_(...).jpg": ["caption1", "caption2", "caption3", "caption4", ..."caption10"], ...

##Dependencies
I have used the following versions for code work:

python==3.8.8
tensorflow==2.4.1
tensorflow-gpu==2.4.1
numpy==1.19.1
h5py==2.10.0

#setting
For my training session, I have get best results with this `lib/config.py` file :
~~~python
num_gpus    = len(tf.config.list_physical_devices('GPU'))
num_workers = num_gpus * 4

class Config:
    data = 'nia'
    data_dir = 'dataset/NIA/'
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
    epoch = 20
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
~~~

##Training
To train the model you need to follow the following steps :

you have to make sure that the training set images are in the folder COCO_dataset/train2014/ and that validation set images are in COCO_dataset/val2014/.
you have to enter all the parameters necessary for the training in the settings.py file.
start the model training with python3 training.py

~~~console
cp lib/config.nia lib/config.py
python run.py train --data=nia

or 

nohup python run.py train --data=nia > train.nia.out &
tail -f train.nia.out
~~~

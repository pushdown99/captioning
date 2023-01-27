# image captioning

keras/tensorflow image captioning application using CNN and transformer as encoder/decoder.<br>
In particulary, the architecture consists of three models:

model|description
---|---
CNN|used to extract the image features. In this application, it used EfficientNetB0 pre-trained on imagenet.
TransformerEncoder|the extracted image features are then passed to a transformer encoder <br>that generates a new representation of the inputs.
TransformerDecoder|it takes the encoder output and the text data sequence as inputs and tries to learn to generate the caption.

(reference github) https://github.com/Dantekk/image-captioning

---
1. [Progress](#Progress)
2. [Dataset](#Dataset) 
3. [Training](#Training)
4. [Evaluate](#Evaluate)
5. [Inference](#Inference)
6. [Result](#Result)
7. [Comparison](#Comparison)
---

## Progress
- [x] guide for project setup
- [x] uploading pretrained model and format-compatible datasets.
- [x] guide for model training
- [x] guide for model evaluation with pretrained model
- [x] guide for model inferencing example

---

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

python: 3.9.10
tensorflow: 2.9.3
cuda: 11.2
cudnn: 8

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

---

## Training
To train the model you need to follow the following steps :

you have to make sure that the training and valid set images are in the folder NIA_dataset/images/ 
you have to enter all the parameters necessary for the training in the config.py file.
start the model training with python run.py train

~~~console
$ cp lib/config.nia lib/config.py
$ python run.py train --data=nia
~~~

~~~console
$ nohup python run.py train --data=nia > train.nia.out &
$ tail -f train.nia.out
~~~
---
## Evaluate
To evaluate the model you need to follow the following steps :

~~~console
$ nohup python run.py eval --data=nia > eval.nia.out &
$ tail -f eval.nia.out
~~~
---
## Inference
To iniference the model you need to follow the following steps :

~~~console
$ nohup python run.py inference --data=nia --sample={sample image}
$ python run.py inference --sample='sample/IMG_0047936_cell_phone.jpg'
Prediction:  휴대폰 이 나무 테이블 위 에 있다
~~~
![](sample/IMG_0047936_cell_phone.jpg)

---
## Result
When you evaluate model with a cleansing dataset,
then show this results.

bleu|result
---|---
BLEU-1|75.91 %
BLEU-2|62.23 %
BLEU-3|50.18 %
BLEU-4|36.37 %

---
## Comparison


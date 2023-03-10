import tensorflow as tf
import numpy as np
import os
import re
import sys
import json
import codecs
import pandas as pd
import time
import datetime
from os.path import join,isfile

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from datetime import datetime as dt
from timeit import default_timer as timer

from tqdm import tqdm
from _pickle import dump,load

from IPython.display import Image, display
import climage

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model, load_img, img_to_array
from lib.config import opt

def is_notebook() -> bool:
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    else:
      return False  # Other type (?)
  except NameError:
    return False    # Probably standard Python interpreter

def Display (file):
  if is_notebook():
    display(Image(file))
  else:
    out = climage.convert(file)
    print (out)

def History (history, path):
  now = time.localtime()
  for label in ["loss","val_loss"]:
    plt.plot(history.history[label],label=label)
  plt.legend()
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.savefig(path)
  plt.show()

@tf.keras.utils.register_keras_serializable()
class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __del__ (self):
    print('[-] custom_schedule deleted.')

  def __init__(self, d_model, warmup_steps=4000):
    super(custom_schedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    config = {
      'd_model': self.d_model,
      'warmup_steps': self.warmup_steps
    }
    return config

class TransformerEncoderBlock (layers.Layer):
  def __del__ (self):
    print('[-] TransformerEncoderBlock deleted.')

  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim   = embed_dim
    self.dense_dim   = dense_dim
    self.num_heads   = num_heads
    self.attention   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.dense_proj  = layers.Dense(embed_dim, activation='relu')
    self.layernorm_1 = layers.LayerNormalization()

  def call (self, inputs, training, mask=None):
    inputs           = self.dense_proj(inputs)
    attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=None)
    proj_input       = self.layernorm_1(inputs + attention_output)

    return proj_input

class PositionalEmbedding (layers.Layer):
  def __del__ (self):
    print('[-] PositionalEmbedding deleted.')

  def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.token_embeddings    = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
    self.sequence_length     = sequence_length
    self.vocab_size          = vocab_size
    self.embed_dim           = embed_dim

  def call(self, inputs):
    length             = tf.shape(inputs)[-1]
    positions          = tf.range(start=0, limit=length, delta=1)
    embedded_tokens    = self.token_embeddings(inputs)
    embedded_positions = self.position_embeddings(positions)

    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask=None):
    return tf.math.not_equal(inputs, 0)

class TransformerDecoderBlock(layers.Layer):
  def __del__ (self):
    print('[-] TransformerDecoderBlock deleted.')

  def __init__(self, embed_dim, ff_dim, num_heads, seq_length, vocab_size, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim   = embed_dim
    self.ff_dim      = ff_dim
    self.num_heads   = num_heads
    self.vocab_size  = vocab_size
    self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.dense_proj  = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
    self.layernorm_1 = layers.LayerNormalization()
    self.layernorm_2 = layers.LayerNormalization()
    self.layernorm_3 = layers.LayerNormalization()
    #self.embedding   = PositionalEmbedding(embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=self.vocab_size)
    self.embedding   = PositionalEmbedding(embed_dim=embed_dim, sequence_length=seq_length, vocab_size=vocab_size)
    self.out         = layers.Dense(self.vocab_size)
    self.dropout_1  = layers.Dropout(0.1)
    self.dropout_2  = layers.Dropout(0.5)
    self.supports_masking = True


  def call (self, inputs, encoder_outputs, training, mask=None):
    inputs      = self.embedding(inputs)
    causal_mask = self.get_causal_attention_mask(inputs)
    inputs      = self.dropout_1(inputs, training=training)

    if mask is not None:
      padding_mask  = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
      combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
      combined_mask = tf.minimum(combined_mask, causal_mask)
    else :
      combined_mask = None
      padding_mask  = None


    attention_output_1 = self.attention_1(
      query=inputs, value=inputs, key=inputs, attention_mask=combined_mask#None
    )
    out_1 = self.layernorm_1(inputs + attention_output_1)

    attention_output_2 = self.attention_2(
      query=out_1,
      value=encoder_outputs,
      key=encoder_outputs,
      attention_mask=padding_mask#None
    )
    out_2 = self.layernorm_2(out_1 + attention_output_2)

    proj_output = self.dense_proj(out_2)
    proj_out    = self.layernorm_3(out_2 + proj_output)
    proj_out    = self.dropout_2(proj_out, training=training)

    preds = self.out(proj_out)
    return preds

  def get_causal_attention_mask(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i    = tf.range(sequence_length)[:, tf.newaxis]
    j    = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype='int32')
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0,)

    return tf.tile(mask, mult)

class ImageCaptioningModel (keras.Model):
  def __del__ (self):
    print('[-] ImageCaptioningModel deleted.')

  def __init__(self, cnn_model, encoder, decoder, num_captions_per_image=5,):
    super().__init__()
    self.cnn_model    = cnn_model
    self.encoder      = encoder
    self.decoder      = decoder
    self.loss_tracker = keras.metrics.Mean(name='loss')
    self.acc_tracker  = keras.metrics.Mean(name='accuracy')
    self.num_captions_per_image = num_captions_per_image

  def call(self, inputs):
    x = self.cnn_model(inputs[0])
    x = self.encoder(x, False)
    x = self.decoder(inputs[2],x,training=inputs[1],mask=None)
    return x

  def calculate_loss(self, y_true, y_pred, mask):
    loss = self.loss(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

  def calculate_accuracy(self, y_true, y_pred, mask):
    accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
    accuracy = tf.math.logical_and(mask, accuracy)
    accuracy = tf.cast(accuracy, dtype=tf.float32)
    mask     = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

  def train_step(self, batch_data):
    batch_img, batch_seq = batch_data
    batch_loss = 0
    batch_acc  = 0
    img_embed  = self.cnn_model(batch_img)

    for i in range(self.num_captions_per_image): # hyhwang: (todo) do not using constants.
      with tf.GradientTape() as tape:
        encoder_out    = self.encoder(img_embed, training=True)
        batch_seq_inp  = batch_seq[:, i, :-1]
        batch_seq_true = batch_seq[:, i, 1:]
        mask           = tf.math.not_equal(batch_seq_inp, 0)
        batch_seq_pred = self.decoder (batch_seq_inp, encoder_out, training=True, mask=mask)
        caption_loss   = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        caption_acc    = self.calculate_accuracy (batch_seq_true, batch_seq_pred, mask)
        batch_loss    += caption_loss
        batch_acc     += caption_acc

      train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)
      grads      = tape.gradient(caption_loss, train_vars)
      self.optimizer.apply_gradients(zip(grads, train_vars))

    loss = batch_loss
    acc  = batch_acc / float(self.num_captions_per_image)

    self.loss_tracker.update_state(loss)
    self.acc_tracker.update_state(acc)

    return {'loss': self.loss_tracker.result(), 'acc': self.acc_tracker.result()}

  def test_step(self, batch_data):
    batch_img, batch_seq = batch_data
    batch_loss = 0
    batch_acc  = 0
    img_embed  = self.cnn_model(batch_img)

    for i in range(self.num_captions_per_image):
      encoder_out    = self.encoder(img_embed, training=False)
      batch_seq_inp  = batch_seq[:, i, :-1]
      batch_seq_true = batch_seq[:, i, 1:]
      mask           = tf.math.not_equal(batch_seq_inp, 0)
      batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=False, mask=mask)
      caption_loss   = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
      caption_acc    = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
      batch_loss    += caption_loss
      batch_acc     += caption_acc

    loss = batch_loss
    acc  = batch_acc / float(self.num_captions_per_image)

    self.loss_tracker.update_state(loss)
    self.acc_tracker.update_state(acc)

    return {'loss': self.loss_tracker.result(), 'acc': self.acc_tracker.result()}

  @property
  def metrics(self):
    # We need to list our metrics here so the `reset_states()` can be
    # called automatically.
    return [self.loss_tracker, self.acc_tracker]

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    strips = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(strips), '')

class TRANSFORMER:
  def __init__ (self, verbose=True):
    self.verbose  = verbose
    self.scores   = list()
    #self.strategy = tf.distribute.MirroredStrategy()
    self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

  def tokenize (self, data, max_vocab_size, seq_length):
    if not isfile (opt.tokenize):
      text_dataset = tf.data.Dataset.from_tensor_slices (data)
      self.TOKENIZER = tf.keras.layers.TextVectorization (
        max_tokens             = max_vocab_size,
        output_mode            = 'int',
        output_sequence_length = seq_length, # input_dim
        standardize            = custom_standardization,
      )
      print ('Start tokenize: lines # ', len (data))
      start = timer ()
      self.TOKENIZER.adapt (text_dataset.batch (1024))
      self.VOCAB_SIZE = len (self.TOKENIZER.get_vocabulary ())
      end   = timer ()
      elapsed = (end-start)
      print ('Elapsed time: ', str (datetime.timedelta (elapsed)))
      dump ({'config': self.TOKENIZER.get_config (), 'weights': self.TOKENIZER.get_weights ()} , open (opt.tokenize, "wb"))
    else:
      token = load (open (opt.tokenize, 'rb'))
      self.TOKENIZER = tf.keras.layers.TextVectorization (
        max_tokens             = token['config']['max_tokens'],
        output_mode            = 'int',
        output_sequence_length = token['config']['output_sequence_length'],
        standardize            = custom_standardization,
      )
      self.TOKENIZER.adapt (tf.data.Dataset.from_tensor_slices (["xyz"]))
      self.TOKENIZER.set_weights (token['weights'])
      self.VOCAB_SIZE = len (self.TOKENIZER.get_vocabulary ())
      print (self.TOKENIZER ("example"))

  def read_image_inf (self, images):
    img = tf.io.read_file (images)
    img = tf.image.decode_jpeg (img, channels=3)
    img = tf.image.resize (img, (299, 299))
    img = tf.image.convert_image_dtype (img, tf.float32)
    img = tf.expand_dims (img, axis=0)
    return img

  def read_image (self, data_augment_flag):
    def decode_image (img_path):
      image = tf.io.read_file (img_path)
      image = tf.image.decode_jpeg (image, channels=3)
      image = tf.image.resize (image, (299, 299))

      if data_augment_flag:
        image = augment (image)
      image = tf.image.convert_image_dtype (image, tf.float32)

      return image

    def augment (image):
      image = tf.expand_dims (image, axis=0)
      image = self.image_transfer (image)
      image = tf.squeeze (image, axis=0)
      return img

    return decode_image

  def make_dataset (self, title, images, captions, data_augment_flag, tokenizer, batch_size, shuffle_dim):
    for i in tqdm (range (1), desc=title):
      AUTOTUNE = tf.data.AUTOTUNE
      read_image_xx   = self.read_image (data_augment_flag)
      image_dataset   = tf.data.Dataset.from_tensor_slices (images)
      image_dataset   = (image_dataset.map (read_image_xx, num_parallel_calls=AUTOTUNE))
      caption_dataset = tf.data.Dataset.from_tensor_slices (captions).map (tokenizer, num_parallel_calls=AUTOTUNE)

      dataset         = tf.data.Dataset.zip ((image_dataset, caption_dataset))
      dataset         = dataset.batch (batch_size).shuffle (shuffle_dim).prefetch (AUTOTUNE)

    return dataset

  def save (self, model, history, len_of_train, len_of_valid, len_of_test, indent = None):
    config = {
      'MAX_VOCAB_SIZE': self.MAX_VOCAB_SIZE,
      'SEQ_LENGTH':     self.SEQ_LENGTH,
      'BATCH_SIZE':     self.BATCH_SIZE,
      'SHUFFLE_DIM':    self.SHUFFLE_DIM,
      'EMBED_DIM':      self.EMBED_DIM,
      'FF_DIM':         self.FF_DIM,
      'NUM_HEADS':      self.NUM_HEADS,
      'EPOCHS':         self.EPOCHS,
      'VOCAB_SIZE':     self.VOCAB_SIZE,
    }
    path = '{}/{}_{}_v{}_{}_{}_{}/'.format('output', dt.now().strftime('%Y%m%d'), self.dataname, self.VOCAB_SIZE, len_of_train, history.history['acc'][0], history.history['val_acc'][0])
    os.makedirs(path, exist_ok=True)

    History (history, path + 'history.png')

    json.dump(history.history, open(path + 'history.json', 'w'), indent=indent)
    json.dump(config, open(path + 'config.json', 'w'), indent=indent)

    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = self.TOKENIZER (input)
    m = tf.keras.Model(input, output)
    m.save(path + 'tokenizer', save_format='tf')
    model.save_weights(path + 'model_weight.h5')

  def get_inference_model(self, path):
    with open(path) as f:
        config = json.load(f)

    EMBED_DIM  = config['EMBED_DIM']
    FF_DIM     = config['FF_DIM']
    NUM_HEADS  = config['NUM_HEADS']
    VOCAB_SIZE = config['VOCAB_SIZE']
    SEQ_LENGTH = config['SEQ_LENGTH']

    cnn_model = EfficientNetB0(input_shape=(*self.IMAGE_SHAPE, 3), include_top=False, weights='imagenet',)
    cnn_model = Model(cnn_model.input, layers.Reshape((-1, 1280))(cnn_model.output))

    encoder = TransformerEncoderBlock (embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS)
    decoder = TransformerDecoderBlock (embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)
    model   = ImageCaptioningModel (cnn_model=cnn_model, encoder=encoder, decoder=decoder)

    ##### It's necessary for init model -> without it, weights subclass model fails
    #cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    cnn_input = tf.keras.layers.Input(shape=(299, 299, 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    model([cnn_input, training, decoder_input])
    #####

    return model

  def generate_caption (self, images, caption_model, tokenizer, SEQ_LENGTH=25):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    img = self.read_image_inf(images)
    img = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img, training=False)

    decoded_caption = 'sos '
    for i in range(max_decoded_sentence_length):
      tokenized_caption = tokenizer([decoded_caption])[:, :-1]
      mask = tf.math.not_equal(tokenized_caption, 0)
      predictions = caption_model.decoder(
        tokenized_caption, encoded_img, training=False, mask=mask
      )
      sampled_token_index = np.argmax(predictions[0, i, :])
      sampled_token = index_lookup[sampled_token_index]
      if sampled_token == 'eos': # end of sentences
        break

      words = decoded_caption.split() # hyhwang, repeated words

      if sampled_token == '??????': # hyhwang, terminal korean word
        decoded_caption += ' ' + sampled_token
        break

      if words[-1] == sampled_token: # repeated word
        x = 1
        #decoded_caption = decoded_caption

      elif words[-1] == 'sos' and sampled_token == 'of': # malformed sentences
        x = 2
        #decoded_caption = decoded_caption

      elif sampled_token != '[UNK]':
        decoded_caption += ' ' + sampled_token

    return decoded_caption.replace('sos ', '')
    #return caption_model


  def inference (self, image, path):
    with open(path + 'config.json') as f:
        config = json.load(f)

    SEQ_LENGTH = config['SEQ_LENGTH']

    tokenizer = tf.keras.models.load_model(path + 'tokenizer')
    tokenizer = tokenizer.layers[1]

    model = self.get_inference_model(path + 'config.json')

    print (path + 'model_weight.h5')
    model.load_weights(path + 'model_weight.h5')

    caption = self.generate_caption (image, model, tokenizer, SEQ_LENGTH)
    print('Prediction: %s' %(caption))
    return caption

  def evaluate (self, path, display = False):
    print ('[+] loading config     :', path + 'config.json')
    with open(join(path,'config.json')) as f:
        config = json.load(f)
        print (config)

    SEQ_LENGTH = config['SEQ_LENGTH']

    print ('[+] loading tokenizer  :', join(path, 'tokenizer'))
    tokenizer = tf.keras.models.load_model(join(path, 'tokenizer'))
    tokenizer = tokenizer.layers[1]

    print ('[+] get inference model:', join(path, 'config.json'))
    model = self.get_inference_model(join(path, 'config.json'))

    print ('[+] loading weight     :', join(path, 'model_weight.h5'))
    model.load_weights(join(path, 'model_weight.h5'))

    print ('[+] loading test data  :', opt.test)
    tests = json.load(codecs.open(opt.test,  'r', 'utf-8-sig'))

    self.bleuc = 0
    self.bleu1 = 0.0
    self.bleu2 = 0.0
    self.bleu3 = 0.0
    self.bleu4 = 0.0

    for k in tqdm(tests):
      actual, predicted = list(), list()
      cap = self.generate_caption (k, model, tokenizer, SEQ_LENGTH)
      inf = [d.split() for d in tests[k]]

      if display:
        Display (k)
        print (k, cap, tests[k])

      predicted.append (cap.split())
      actual.append (inf)

      self.calculate_scores(k, actual, predicted)

    print ('BLEU-1: {:.2f} %'.format(self.bleu1/self.bleuc))
    print ('BLEU-2: {:.2f} %'.format(self.bleu2/self.bleuc))
    print ('BLEU-3: {:.2f} %'.format(self.bleu3/self.bleuc))
    print ('BLEU-4: {:.2f} %'.format(self.bleu4/self.bleuc))

    json.dump(self.scores, open(join(opt.data_dir, 'scores.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

  def fit (self, num_of_epochs = 5):
    self.EPOCHS = num_of_epochs

    # Load dataset
    descriptions =  json.load(codecs.open(opt.captions, 'r', 'utf-8-sig'))
    trains       =  json.load(codecs.open(opt.train,    'r', 'utf-8-sig'))
    valids       =  json.load(codecs.open(opt.val,      'r', 'utf-8-sig'))
    tests        =  json.load(codecs.open(opt.test,     'r', 'utf-8-sig'))
    texts        =  json.load(codecs.open(opt.text,     'r', 'utf-8-sig'))

    self.image_transfer = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomContrast    (factor = (0.05, 0.15)),
      tf.keras.layers.experimental.preprocessing.RandomTranslation (height_factor = (-0.10, 0.10), width_factor = (-0.10, 0.10)),
      tf.keras.layers.experimental.preprocessing.RandomZoom        (height_factor = (-0.10, 0.10), width_factor = (-0.10, 0.10)),
      tf.keras.layers.experimental.preprocessing.RandomRotation    (factor = (-0.10, 0.10))
    ])

    self.tokenize (texts, self.MAX_VOCAB_SIZE, self.SEQ_LENGTH)

    for k in trains:
      if len(trains[k]) != opt.n_caption:
        print ('caption data error', opt.n_caption, k, trains[k])

    ds_train = self.make_dataset('train', list(trains.keys()), list(trains.values()), False, self.TOKENIZER, self.BATCH_SIZE, self.SHUFFLE_DIM)
    ds_valid = self.make_dataset('valid', list(valids.keys()), list(valids.values()), False, self.TOKENIZER, self.BATCH_SIZE, self.SHUFFLE_DIM)
    ds_test  = self.make_dataset('test ', list(tests.keys()),  list(tests.values()),  False, self.TOKENIZER, self.BATCH_SIZE, self.SHUFFLE_DIM)

    # with self.strategy.scope():
    encoder    = TransformerEncoderBlock (embed_dim=self.EMBED_DIM, dense_dim=self.FF_DIM, num_heads=self.NUM_HEADS)
    decoder    = TransformerDecoderBlock (embed_dim=self.EMBED_DIM, ff_dim=self.FF_DIM, num_heads=self.NUM_HEADS, seq_length=self.SEQ_LENGTH, vocab_size=self.VOCAB_SIZE)
    model      = ImageCaptioningModel    (self.cnn_model, encoder, decoder)

    cross_entropy  = SparseCategoricalCrossentropy (from_logits=True, reduction='none')
    #early_stopping = EarlyStopping (patience=3, restore_best_weights=True)
    early_stopping = EarlyStopping (monitor='val_accuracy', patience=8, min_delta=0.001, restore_best_weights=True, mode='max')
    lr_scheduler   = custom_schedule (self.EMBED_DIM)
    optimizer      = Adam (learning_rate=lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile (optimizer=optimizer, loss=cross_entropy)

    history = model.fit (
      ds_train, 
      epochs=num_of_epochs, 
      validation_data=ds_valid, 
      callbacks=[early_stopping],
      verbose = 1,
    )
    self.stopped_epoch = early_stopping.stopped_epoch
    print ('stopped_epoch: ', self.stopped_epoch)

    # Compute definitive metrics on train/valid set
    mt_train = model.evaluate (ds_train, batch_size=self.BATCH_SIZE)
    mt_valid = model.evaluate (ds_valid, batch_size=self.BATCH_SIZE)
    mt_test  = model.evaluate (ds_test,  batch_size=self.BATCH_SIZE)

    print ('Train Loss = %.4f - Train Accuracy = %.4f' % (mt_train[0], mt_train[1]))
    print ('Valid Loss = %.4f - Valid Accuracy = %.4f' % (mt_valid[0], mt_valid[1]))
    print ('Test  Loss = %.4f - Test  Accuracy = %.4f' % (mt_test[0],  mt_test[1]) )

    self.save (model, history, len (trains), len (valids), len (tests), 2)

  # Calculate BLEU score
  def calculate_scores (self, path, actual, predicted, display = False):
    if ' '.join (predicted[0]) == '': return

    smooth = SmoothingFunction ().method4
    bleu1 = corpus_bleu (actual, predicted, weights=(1.0, 0, 0, 0),           smoothing_function=smooth)*100
    bleu2 = corpus_bleu (actual, predicted, weights=(0.5, 0.5, 0, 0),         smoothing_function=smooth)*100
    bleu3 = corpus_bleu (actual, predicted, weights=(0.3, 0.3, 0.3, 0),       smoothing_function=smooth)*100
    bleu4 = corpus_bleu (actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)*100

    if display:
      print ('BLEU-1: %f' % bleu1)
      print ('BLEU-2: %f' % bleu2)
      print ('BLEU-3: %f' % bleu3)
      print ('BLEU-4: %f' % bleu4)

    self.bleuc += 1
    self.bleu1 += bleu1
    self.bleu2 += bleu2
    self.bleu3 += bleu3
    self.bleu4 += bleu4

    score = dict ()
    score['path']       = path
    score['bleu1']      = bleu1
    score['predicted']  = ' '.join (predicted[0])
    score['actual']     = list ([' '.join (d) for d in actual[0]])
    self.scores.append (score)


#  # evaluate the skill of the model
#  def evaluate_model (self, model, descriptions, features, tokenizer, max_length):
#    actual, predicted = list(), list()
#    # step over the whole set
#    for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
#      # generate description
#      yhat = self.generate_desc(model, tokenizer, features[key], max_length)
#      # store actual and predicted
#      references = [d.split() for d in desc_list]
#      actual.append(references)
#      predicted.append(yhat.split())
#    print('Sampling:')
#    self.calculate_scores(path, actual, predicted)

#  def evaluate (self, filename, max_length):
#    test = load(open('data/{}_test.pkl'.format(self.dataname), 'rb'))
#    test_descriptions = load(open('data/{}_test_descriptions.pkl'.format(self.dataname), 'rb'))
#    tokenizer = load(open('data/{}_tokenizer.pkl'.format(self.dataname), 'rb'))
#
#    test_features = self.load_photo_features('data/rnn_vgg16_{}.pkl'.format(self.dataname), test)
#    print('Photos: test=%d' % len(test_features))
#
#    # load the model
#    model = tf.keras.models.load_model (filename)
#    self.evaluate_model (model, test_descriptions, test_features, tokenizer, max_length)


class efficientnetb0 (TRANSFORMER):
  def __del__ (self):
    print('[-] efficientnetb0 deleted.')

  def __init__ (self, verbose=True):
    super().__init__(verbose)

    self.dataname       = opt.data
    self.IMAGE_SHAPE    = opt.IMAGE_SHAPE
    self.MAX_VOCAB_SIZE = opt.MAX_VOCAB_SIZE
    self.SEQ_LENGTH     = opt.SEQ_LENGTH
    self.BATCH_SIZE     = opt.BATCH_SIZE
    self.SHUFFLE_DIM    = opt.SHUFFLE_DIM
    self.EMBED_DIM      = opt.EMBED_DIM
    self.FF_DIM         = opt.FF_DIM
    self.NUM_HEADS      = opt.NUM_HEADS

    #with self.strategy.scope():
    model          = EfficientNetB0 (input_shape=(*self.IMAGE_SHAPE, 3), include_top=False, weights='imagenet',)
    model          = Model (model.input, layers.Reshape((-1, 1280))(model.output))
    self.cnn_model = model


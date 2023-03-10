from IPython.display import Image, display

import numpy as np
import matplotlib.pyplot as plt
import climage
from keras.utils import plot_model, load_img, img_to_array

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
    out = climage.convert(file, is_unicode=True)
    print (out)

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (299, 299))
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def plot(model, filename):
  plot_model(model, to_file=filename, show_shapes=True, show_layer_names=False)
  display(Image(filename))

def History (history):
  for label in ["loss","val_loss"]:
    plt.plot(history.history[label],label=label)
  plt.legend()
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.savefig('models/rnn_2_vgg16_loss.png')
  plt.show()

def plot_loss(loss_plot):
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.show()    
    
def plot_attention(image, result, attention_plot):
  temp_image = np.array(Image.open(image))

  fig = plt.figure(figsize=(10, 10))

  len_result = len(result)
  for l in range(len_result):
    temp_att = np.resize(attention_plot[l], (8, 8))
    ax = fig.add_subplot(len_result//2, len_result//2, l+1)
    ax.set_title(result[l])
    img = ax.imshow(temp_image)
    ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

  plt.tight_layout()
  plt.show()

class CSVLog:
  """
  Logs to a CSV file.
  """
  def __init__(self, filename):
    self._filename = filename
    self._header_written = False

  def log(self, items):
    keys = list(items.keys())
    file_mode = "a" if self._header_written else "w"
    with open(self._filename, file_mode) as fp:
      if not self._header_written:
        fp.write(",".join(keys) + "\n")
        self._header_written = True
      values = [ str(value) for (key, value) in items.items() ]
      fp.write(",".join(values) + "\n")

class BestWeightsTracker:
  def __init__(self, filepath):
    self._filepath = filepath
    self._best_weights = None
    self._best_mAP = 0

  def on_epoch_end(self, model, mAP):
    if mAP > self._best_mAP:
      self._best_mAP = mAP
      self._best_weights = model.get_weights()

  def restore_and_save_best_weights(self, model):
    if self._best_weights is not None:
      model.set_weights(self._best_weights)
      model.save_weights(filepath = self._filepath, overwrite = True, save_format = "h5")
      print("Saved best model weights (Mean Average Precision = %1.2f%%) to '%s'" % (self._best_mAP, self._filepath))

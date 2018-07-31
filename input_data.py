"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib

import numpy
import numpy as np

import sys
if (sys.version_info > (3, 0)):
    from functools import reduce
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


SOURCE_DIGITS = 'http://yann.lecun.com/exdb/mnist/'
SOURCE_FASHION = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'


def maybe_download(filename, work_directory, SOURCE_URL):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, n_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * n_classes
  labels_one_hot = numpy.zeros((num_labels, n_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

  
def convert_one_hot(labels, dim):
    n = len(labels)
    targets = np.zeros((n, dim))
    targets[np.arange(n), labels] = 1
    return targets


def extract_labels(filename):#, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    #if one_hot:return dense_to_one_hot(labels)
    return labels

from operator import mul


class DataSet(object):

  def __init__(self, images, labels, n_classes, one_hot, flatten=True):
    assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if flatten: images = images.reshape(images.shape[0], reduce(mul, images.shape[1:], 1))
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(numpy.float32)
    self._images = images
    self._labels = convert_one_hot(labels, n_classes) if one_hot else labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      #print (batch_size , self._num_examples)
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled, n_classes, one_hot, flatten=True):
        #self.n_unlabeled = n_unlabeled
        num_examples = len(labels)

        if n_labeled>0:
            # Labeled DataSet
            self.n_labeled = n_labeled
            indices = numpy.arange(num_examples)
            shuffled_indices = numpy.random.permutation(indices)
            images = images[shuffled_indices]
            labels = labels[shuffled_indices]
            
            n_from_each_class = int(n_labeled / n_classes)
            i_labeled = []
            for c in range(n_classes):
                i = indices[labels==c][:n_from_each_class]
                i_labeled += list(i)
            l_images = images[i_labeled]
            l_labels = labels[i_labeled]
            self.labeled_ds = DataSet(l_images, l_labels, n_classes, one_hot=one_hot, flatten=flatten)
        
        else:
            self.n_labeled = n_labeled
            self.labeled_ds = DataSet(images, labels, n_classes, one_hot=one_hot, flatten=flatten)
        
        self.unlabeled_ds = DataSet(images, labels, n_classes, one_hot=one_hot, flatten=flatten)


    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return labeled_images, labels, unlabeled_images

def read_mnist(train_dir, n_labeled=-1, one_hot=False, SOURCE_URL=SOURCE_DIGITS):
  class DataSets(object): pass
  data_sets = DataSets()

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 0

  local_file = maybe_download(TRAIN_IMAGES, train_dir, SOURCE_URL)
  train_images = extract_images(local_file)

  local_file = maybe_download(TRAIN_LABELS, train_dir, SOURCE_URL)
  train_labels = extract_labels(local_file)#, one_hot=one_hot)

  local_file = maybe_download(TEST_IMAGES, train_dir, SOURCE_URL)
  test_images = extract_images(local_file)

  local_file = maybe_download(TEST_LABELS, train_dir, SOURCE_URL)
  test_labels = extract_labels(local_file)#, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  n_classes = np.amax(train_labels)+1
  data_sets.train = SemiDataSet(train_images/255., train_labels, n_labeled, n_classes, one_hot=one_hot)
  data_sets.validation = DataSet(validation_images/255., validation_labels, n_classes, one_hot=one_hot)
  data_sets.test = DataSet(test_images/255., test_labels, n_classes, one_hot=one_hot)

  return data_sets

  
from sklearn import datasets
  
def make_half_moons(n_training, n_test, noise=None):
    data, labels = datasets.make_moons(n_samples=n_training+n_test, shuffle=True, noise=noise, random_state=None)
    return data[:n_training], labels[:n_training], data[n_training:], labels[n_training:]
    
    
def make_moons(n_labeled, n_unlabeled, n_test, one_hot=True, noise=None):
    class DataSets(object): pass
    data_sets = DataSets()

    train_images, train_labels, test_images, test_labels = make_half_moons(n_labeled+n_unlabeled, n_test)

    n_classes = np.amax(train_labels)+1
    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled, n_unlabeled, n_classes, one_hot=one_hot)
    data_sets.test = DataSet(test_images, test_labels, n_classes, one_hot=one_hot)
    
    #plt.scatter(d1[:,0], d1[:,1], c=l1)
    #plt.show()
    return data_sets


import pickle

# Function to load a batch into memory
def load_batch(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    feats = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    lbls = batch['labels']
    return feats, lbls


def normalize(x):
    return x / 255.


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer().fit(range(10))
def one_hot_encode(x):
    global lb
    return lb.transform(x)

cifar10_dir='cifar-10-batches-py'

def read_cifar10(data_path, one_hot=1):
    class DataSets(object): pass
    data_sets = DataSets()

    ffs, lls = [], []

    data_dir = os.path.join(data_path, cifar10_dir)
    for i in range(1, 6):
        feats, lbls = load_batch(data_dir, 'data_batch_%i' % i)
        norm_feats = normalize(feats)
        ffs.append(norm_feats)
        lls.append(lbls)

    train_images = np.concatenate(ffs)
    train_labels = np.array([item for sublist in lls for item in sublist])

    n_labeled = -1
    n_classes = np.amax(train_labels)+1
    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled, n_classes, one_hot=one_hot, flatten=False)

    test_images, test_labels = load_batch(data_dir, 'test_batch')
    test_images = normalize(test_images)
    test_labels = np.array(test_labels)
    data_sets.test = DataSet(test_images, test_labels, n_classes, one_hot=one_hot, flatten=False)

    #print(X_train.shape, R_train.shape)
    #plt.scatter(d1[:,0], d1[:,1], c=l1)
    #plt.show()
    return data_sets

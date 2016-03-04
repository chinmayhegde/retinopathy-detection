import gzip
import math
import os
import numpy as np
import cv2
import csv
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):

    dtype = tf.as_dtype(dtype).base_dtype

    #Reshape images/labels
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1])

    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = len(images)

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

  def next_batch(self, batch_size, fake_data=False):
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
      assert batch_size <= self._num_examples

    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  name_label_association = read_labels()
  all_images, all_labels = read_images(train_dir, name_label_association)

  TRAIN_SIZE = (len(all_images) * 3 ) / 4 #3/4 train
  TEST_SIZE = len(all_images) / 8 #1/8 test
  VALIDATION_SIZE = len(all_images) / 8 #1/8 validation

  train_images = np.asarray(all_images[:TRAIN_SIZE]) #Grab training images
  test_images = np.asarray(all_images[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]) #Grab test images
  validation_images = np.asarray(all_images[TRAIN_SIZE + TEST_SIZE:]) #Grab validation images

  train_labels= np.asarray(all_labels[:TRAIN_SIZE]) #Grab training images
  test_labels = np.asarray(all_labels[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]) #Grab test images
  validation_labels = np.asarray(all_labels[TRAIN_SIZE + TEST_SIZE:]) #Grab validation images

  data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
  data_sets.test = DataSet(test_images, test_labels, dtype=dtype)
  data_sets.validation = DataSet(validation_images, validation_labels, dtype=dtype)
  data_sets._num_examples = len(all_images)

  return data_sets

def read_images(directory, name2label):
  images = []
  labels = []

  # TODO: we should experiment with these HOG parameters
  win_size = (16, 16)
  block_size = (16, 16)
  block_stride = (8, 8)
  cell_size = (8, 8)
  num_bins = 9
  hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

  #KMeans parameters
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.5) #Stop at 80% accuracy? or after 10 iterations
  k = 1024 #Number of centroids to find

  imgDir = os.listdir(directory)

  index = 0

  for image_name in imgDir:
    image = cv2.imread(os.path.join(directory, image_name), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    #image = hog.compute(image)
    #print len(image)
    #ret, label, center = cv2.kmeans(image, k, criteria, 10, 0)
    #print len(center)

    if (index % 250) == 0:
      print "Loaded image", index

    images.append(image)

    #One-Hot vector
    label = numpy.zeros(5)
    label[name2label[image_name.split('.')[0]]] = 1
    labels.append(label)

    index = index + 1

  return images, labels

def read_labels():
  # Get image names and classifications
  imageLabelAssociation = {}

  with open('subsetcsv.csv', 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        imageLabelAssociation[row[0]] = row[1]

  return imageLabelAssociation


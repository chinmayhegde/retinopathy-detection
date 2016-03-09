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

  def __init__(self, name_label_association, fake_data=False, one_hot=False,
               dtype=tf.float32):

    dtype = tf.as_dtype(dtype).base_dtype

    self._images = [] #Images loaded in batching
    self._labels = name_label_association 
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = len(name_label_association)

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

    #shuffle if necessary
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples


    end = self._index_in_epoch

    #Get images, labels for given range #TODO: This is disgusting. Better way?
    label_image_association_to_load = {}
    index = 0

    print "Preparing batch of size", batch_size

    for key, value in self._labels.items():
      if index >= start and index <= end:
        label_image_association_to_load[key] = value

    images , labels = read_given_images_and_labels(label_image_association_to_load)

    #Reshape images/labels
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    labels = labels.reshape(labels.shape[0], labels.shape[1])

    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)

    return images, labels


def read_data_sets(train_dir, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  name_label_association = read_labels(train_dir)

  TRAIN_SIZE = (len(name_label_association) * 3 ) / 4 #3/4 train
  TEST_SIZE = len(name_label_association) / 8 #1/8 test
  VALIDATION_SIZE = len(name_label_association) / 8 #1/8 validation

  train_label_image_association = {}
  test_label_image_association = {}
  validation_label_image_association = {}

  index = 0
  for key, value in name_label_association.items():
    if index < TRAIN_SIZE:
      train_label_image_association[key] = value
    elif index > TRAIN_SIZE and index < TRAIN_SIZE + TEST_SIZE:
      test_label_image_association[key] = value
    elif index > TRAIN_SIZE + TEST_SIZE:
      validation_label_image_association[key] = value

    index = index + 1

  data_sets.train = DataSet(train_label_image_association, dtype=dtype)
  data_sets.test = DataSet(test_label_image_association, dtype=dtype)
  data_sets.validation = DataSet(validation_label_image_association, dtype=dtype)
  data_sets._num_examples = len(name_label_association)

  return data_sets

#Used to read a given subset of images
def read_given_images_and_labels(name2label):
  images = labels = []

  for label , image in name2label:
    image = cv2.imread(os.path.join(directory, image), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image (512, 512))

    images.append(image)

    #One-Hot vector
    label = numpy.zeros(5)
    label[name2label[image_name.split('.')[0]]] = 1
    labels.append(label)

  return np.asarray(images), np.asarray(labels)

def read_labels(train_dir):
  # Get image names and classifications
  imageLabelAssociation = {}

  with open('../../../trainLabels.csv', 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        imageLabelAssociation[row[0]] = train_dir + '/' + row[1] #Label = imagepath

  return imageLabelAssociation


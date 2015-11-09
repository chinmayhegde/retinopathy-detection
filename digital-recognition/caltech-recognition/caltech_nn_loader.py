"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip
import os
import cv2
import numpy
import random

# Third-party libraries
import numpy as np

def load_data_wrapper():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	categories_path = os.path.join(dir_path, '101_ObjectCategories')
	categories = os.listdir(categories_path)

	train_data = []
	train_vectors = []
	val_data = []
	val_labels = []
	test_data = []
	test_labels = []

	label = 0
	for i in random.sample(range(len(categories)), 4): # pick four random categories
		category = categories[i]
		image_path = os.path.join(categories_path, category)
		image_list = os.listdir(image_path)
		image_count = 0

		for image_name in image_list:
			image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (300, 200))
			flat = image.flatten().astype(numpy.float32)
			# first 20 go in testing
			if image_count < 20:
				train_data.append(flat)
				train_vectors.append(vectorized_result(label))
			#Next 9 in validation
			elif image_count > 20 < 30:
				val_data.append(flat)
				val_labels.append([label])
			#last in testing
			elif image_count < 40:
				test_data.append(flat)
				test_labels.append([label])
			image_count += 1
			

	train_data = numpy.array(train_data)
	train_vectors = numpy.array(train_vectors)
	training_data = zip(train_data, train_vectors)

	val_data = numpy.array(val_data)
	val_labels = numpy.array(val_labels)
	validation_data = zip(val_data, val_labels)

	test_data = numpy.array(test_data)
	test_labels = numpy.array(test_labels)
	test_data = zip(test_data, test_labels)
	return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((4, 1))
    e[j] = 1.0
    return e
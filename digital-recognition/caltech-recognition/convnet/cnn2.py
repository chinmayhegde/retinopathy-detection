from __future__ import division
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from scipy.signal import convolve2d, correlate2d
from layers2 import InputLayer, FullyConnectedLayer, ReLuLayer, DropoutLayer, \
                   ConvolutionLayer, PoolingLayer, SquaredLossLayer, SoftmaxLossLayer

import numpy as np
import os
import cv2
import random
import sys
import pickle

class NeuralNet:
    def __init__(self, layers, l2_decay=0.001, debug=False, learning_rate=0.001):
        mapping = {"input": lambda x: InputLayer(x),
                   "fc": lambda x: FullyConnectedLayer(x),
                   "convolution": lambda x: ConvolutionLayer(x),
                   "pool": lambda x: PoolingLayer(x),
                   "squaredloss": lambda x: SquaredLossLayer(x),
                   "softmax": lambda x: SoftmaxLossLayer(x),
                   "relu": lambda x: ReLuLayer(x),
                   "dropout": lambda x: DropoutLayer(x)}
        self.layers = []
        self.l2_decay = l2_decay
        self.debug = debug
        self.learning_rate = learning_rate
        prev = None

        np.seterr(all="warn")
        
        #print str(layers)

        for layer in layers:
            layer["input_shape"] = layer.get("input_shape", None) or prev.output_shape
            layer["l2_decay"] = layer.get("l2_decay", None) or self.l2_decay
            layer["debug"] = self.debug
            layer = mapping[layer["type"]](layer)
            self.layers.append(layer)
            prev = layer

    def forward(self, input):
        inputs = [input]

        for layer in self.layers:
            #print str(layer.input_shape)
            #print str(inputs[-1].shape)
            assert(layer.input_shape == inputs[-1].shape)
            inputs.append(layer.forward(inputs[-1]))

        return inputs

    def backward(self, inputs, parent_gradient):
        gradients = [parent_gradient]        
        
        for input, layer in zip(inputs[:-1][::-1], self.layers[::-1]):
            gradients.append(layer.backward(input, gradients[-1]))

        return gradients

    def update(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def loss(self, input, expected):
        prediction = self.predict(input)
        loss = self.layers[-1].loss(prediction, expected)

        for layer in self.layers[:-1][::-1]:
            loss += layer.loss() # regularization terms

        return loss

    def predict(self, buffer):
        inputs = [buffer]

        for layer in self.layers:
            assert(layer.input_shape == inputs[-1].shape)
            inputs.append(layer.predict(inputs[-1]))

        return inputs[-1]
        
    def load_data(self, filename="layers.pickle"):
        file = open(filename, "rb")
        self.layers = pickle.load(file)
        file.close()
        
    def save_data(self, filename="layers.pickle"):    
        file = open("layers.pickle", "wb")
        pickle.dump(self.layers,file)
        file.close()

    def train(self, X, y, n_epochs=10, n_samples=None):
        curr_epoch = 0
        print str(len(X))
        for epoch in range(0, n_epochs):
            print str(curr_epoch) + " " + str(datetime.now())
            curr_epoch += 1
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            for i in indices[:n_samples]:
                self._train(X[i], y[i])

    def _train(self, x, y):
        inputs = self.forward(x)
        #print str(len(inputs))
        gradients = self.backward(inputs, y)
        if self.debug:
            numerical = self.numerical_gradient(x, y)
            if not np.all(abs(numerical - gradients[-1]) < 0.00001):
                print "Numerical gradient:\n {}\nAnalytical gradient:\n {}".format(numerical, gradients[-1])
                print "loss: {}\n".format(self.loss(x, y))
                assert(False)

        self.update(self.learning_rate)

    def numerical_gradient(self, input, expected):
        eps = 0.000001
        pert = input.copy()
        res = np.zeros(shape=input.shape)

        for index, x in np.ndenumerate(input):
            pert[index] = input[index] + eps
            res[index] = (self.loss(pert, expected) - self.loss(input, expected))/eps
            pert[index] = input[index]

        return res

def _error(X, y):
    assert(len(X) == len(y))
    mispred = 0
    print "number of images testing " + str(len(X))
    for i in range(0, len(X)):
        temp = net.predict(X[i])
        prediction = np.argmax(temp)
        if prediction != y[i]:
            print ">>>" + str(i) + "Label " + str(y[i]) + " prediction " + str(prediction) + " " + str(temp)
            cv2.imshow(str(temp), X[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        else:
            print str(i) + "Label " + str(y[i]) + " prediction " + str(prediction) + " " + str(temp)
        mispred += prediction != y[i]
    return 100*mispred/len(X)

if __name__ == "__main__":
    print "\nstart " + str(datetime.now()) + "\n"
    n_classes = 4
    net = NeuralNet([{"type": "input", "input_shape": (200, 200)},
                     {"type": "convolution", "filters": 5, "size": 3},
                     {"type": "dropout"},
                     {"type": "relu"},
                     {"type": "pool", "size": 2},
                     {"type": "fc", "neurons": 100},
                     {"type": "dropout"},
                     {"type": "relu"},
                     {"type": "fc", "neurons": n_classes},
                     {"type": "relu"},
                     {"type": "softmax", "categories": n_classes}])
                     
    digits = load_digits(n_class=n_classes)
    #print str(digits.images)
    #print str(digits.images.shape)
    #print str(digits.images.reshape(len(digits.images), 64))
    #data2 = .reshape(len(digits.images), 64)
    #print str(digits.target)

    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    categories_path = os.path.join(dir_path, '101_ObjectCategories')
    categories = os.listdir(categories_path)
    data_images = []
    labels = []
    hog = cv2.HOGDescriptor()
    counter = 0
    trained_indices = [40,88,31,67]
    #random.sample(range(len(categories)), 4)
    for i in range(4): # pick four random categories
        category = categories[trained_indices[i]]
        print category + " " + str(counter)
        image_path = os.path.join(categories_path, category)
        image_list = os.listdir(image_path)
        image_count = 0
        
        for image_name in image_list:
            image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
            #if image_count == 0:
                #print str(image)
                #print str(image.shape)
            image = cv2.resize(image, (200, 200))
            #if image_count == 0:
                #print "resized\n" + str(image)
                #print str(image.shape)                
            #image = hog.compute(image)
            #if image_count == 0:
                #print "hog\n" + str(image)
                #print str(image.shape)
            #flat = image.flatten().astype(np.float32)
            #data_images.append(flat)
            data_images.append(image)
            labels.append(counter)
            image_count += 1
        counter += 1
    
    data_images = np.array(data_images);
    labels = np.array(labels)
    #print str(data_images.shape)

    data = data_images.reshape(len(data_images), 40000)
    #train_data, test_data, train_target, test_target = train_test_split(digits.images.reshape(len(digits.images), 64), digits.target, train_size=0.9)
    train_data, test_data, train_target, test_target = train_test_split(data, labels, train_size=0.5)
    #print str(train_target)
    train_data = train_data.reshape((len(train_data), 200, 200))
    test_data = test_data.reshape((len(test_data), 200, 200))
    #train_data = train_data.reshape((len(train_data), 8, 8))
    #test_data = test_data.reshape((len(test_data), 8, 8))
    
    if len(sys.argv) == 2:
        for j in range(int(sys.argv[1])):
            net.train(train_data, train_target, n_epochs=1)
            print "Epoch " + str(j) + ": Test Error {:.2f}%".format(_error(train_data, train_target))
            print "Epoch " + str(j) + ": Test Error {:.2f}%".format(_error(test_data, test_target))
            
    elif len(sys.argv) == 3:
        net.load_data(filename=sys.argv[2])
        print "Epoch " + str(49) + ": Train Error {:.2f}%".format(_error(train_data, train_target))
        print "Epoch " + str(49) + ": Test Error {:.2f}%".format(_error(test_data, test_target))
    elif len(sys.argv) == 4:
        net.load_data(sys.argv[2])
        net.train(train_data, train_target, n_epochs=int(sys.argv[1]))
    else:
        net.train(train_data, train_target, n_epochs=10)
    
    net.save_data()
    #print "Train Error: {:.2f}%".format(_error(train_data, train_target))
    #print "Test Error: {:.2f}%".format(_error(test_data, test_target))
    print "\nend " + str(datetime.now()) + "\n"
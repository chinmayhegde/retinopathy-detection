from __future__ import division
from datetime import datetime
from sklearn.cross_validation import train_test_split
from scipy.signal import convolve2d, correlate2d
from layers import InputLayer, FullyConnectedLayer, ReLuLayer, DropoutLayer, \
                   ConvolutionLayer, PoolingLayer, SquaredLossLayer, SoftmaxLossLayer

import numpy as np
import os
import cv2
import random
import sys
import pickle
import csv
import argparse


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
            '''if prediction != y[i]:
                print ">>> [" + str(i) + "] Label " + str(y[i]) + " prediction " + str(prediction) + " " + str(temp)
            else:
                print "[" + str(i) + "] Label " + str(y[i]) + " prediction " + str(prediction) + " " + str(temp)'''
            mispred += prediction != y[i]
        return 100*mispred/len(X)
        
def is_valid_file(arg):
    if not os.path.exists(arg):
        print "invalid file " + arg
        sys.exit()
        
def get_label(n_classes, classification):
    if n_classes==2:
        if classification==0:
            return 0
        else:
            return 1
    if n_classes==3:
        if classification==0:
            return 0
        elif classification==2:
            return 1
        else:
            return 2
    if n_classes==4:
        if classification==0:
            return 0
        elif classification==2:
            return 1
        elif classification==3:
            return 2
        else:
            return 3
    if n_classes==5:
        return classification
        
#params
#   image: grayscale image before resize
#   ratio: decimal value giving the portion of the vertical radius to keep
#   resizeHeight/resizeWidth: int values to resize the image to after
#return 
#   (cropped image, success boolean)
def crop_img(image, resizeHeight= 518, resizeWidth=718, ratio=.75):
    
    #reduce size proportionally
    div=3
    height, width = image.shape
    image = cv2.resize(image, (int(round(width/div)), int(round(height/div))))
    img = image.copy()
    
    passed = False
    
    #Some of the images are darker than others so it is necessary to loop through
    #various threshold values until a proper bounding rectangle is found
    for i in range(0,15):
        ret,thresh = cv2.threshold(img,10+(5*i),255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        brx,bry,brw,brh = cv2.boundingRect(cnt)
        if(brw > 100 and brh > 100):
            passed = True
            break
        else:
            #blur if needed (save time not blurring every image, only a few need it)
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
    if not passed:
        print "crop failed"
        return (image, False)
    
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)

    height, width = image.shape
    newY = max(int(y-(radius*ratio)), 0)
    newHeight = int(radius*ratio*2)
    if newY + newHeight > height:
        newHeight = height - newY
    
    #NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    crop_img = image[newY:newY+newHeight, brx:brx+brw]
    crop_img = cv2.resize(crop_img, (resizeWidth, resizeHeight))
    return (crop_img, True)

if __name__ == "__main__":
    print "\nstart " + str(datetime.now()) + "\n"
    
    #command line parsing    
    parser = argparse.ArgumentParser(description="diabetic retinopothy cnn")
    parser.add_argument("-p", dest="pklfile", help="name of pkl file to run with", metavar="FILE")
    parser.add_argument("-i", "--iteration", type=int, help="number of epochs to test, one epoch at a time")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs to train")
    parser.add_argument("-t", dest="train_path", help="path to train folder with images", metavar="FILE", default="../../../../train/")#train\\
    parser.add_argument("-c", dest="label_file", help="csv file of labels", metavar="FILE", default="trainLabels.csv")
    parser.add_argument("-r", "--results", action="store_true", help="bool run tests")
    parser.add_argument("-n", dest="n_classes", type=int, help="number of classifications", default=2, choices=[2,3,4,5])
    parser.add_argument("-s", dest="size", type=int, help="dimension of image", default=500)
    
    args = parser.parse_args()
    
    #check files
    is_valid_file(args.label_file)
    is_valid_file(args.train_path)
    if not args.pklfile is None:
        is_valid_file(args.pklfile)
        print "pickle not found"
        net.load_data(args.pklfile)
    
    net = NeuralNet([{"type": "input", "input_shape": (size, size)},
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
                     
    
    # Get image names and classifications
    names = {i: [] for i in range(5)}
    with open(args.label_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            names[int(row[1])].append(args.train_path + row[0] + '.jpeg')

    # TODO: we should experiment with these
    '''win_size = (16, 16)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                            num_bins)'''

    # Load all images into memory for now
    #images = {i: [] for i in range(5)}
    counter = 0
    for classification, image_names in names.iteritems():
        for image_name in image_names:
            if (n_classes==2 and classification in [1, 4]) or (n_classes==3 and classification in [0,2,4]) or (n_classes==4 and classification in [0,2,3,4]) or n_classes==5:
                image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
                img, bool = crop_img(image, size, size)
                if(not bool):
                    print image_name + " failed crop"
                    sys.exit()
                #image = hog.compute(image)
                #images[classification].append(image)
                images[classification].append(image)
                data_images.append(img)
                counter = counter + 1
                labels.append(get_label(n_classes, classification))
                if counter%100 == 0:
                    print "image count: " + str(counter)
            

    # Partition images into test and train sets 
    data_images = np.array(data_images);
    labels = np.array(labels)

    data = data_images.reshape(len(data_images), size*size)
    train_data, test_data, train_target, test_target = train_test_split(data, labels, train_size=0.75)
    train_data = train_data.reshape((len(train_data), size, size))
    test_data = test_data.reshape((len(test_data), size, size))
    
    '''train_ratio = 0.75
    train_labels = []
    train_data = []
    test_labels = []
    test_data = []

    for classification, image_list in images.iteritems():
        train_num = int(len(image_list) * train_ratio)
        train_labels.extend([classification for _ in range(train_num)])
        train_data.extend(image_list[:train_num])
        test_labels.extend([classification for _ in
                            range(len(image_list) - train_num)])
        test_data.extend(image_list[train_num:])

    train_labels = numpy.array(train_labels)
    train_data = numpy.array(train_data)
    test_labels = numpy.array(test_labels)
    test_data = numpy.array(test_data)'''
    
    
    if not args.iteration is None:
        for j in range(args.iteration):
            net.train(train_data, train_target, n_epochs=1)
            print "Epoch " + str(j) + ": Train Error {:.2f}%".format(_error(train_data, train_target))
            print "Epoch " + str(j) + ": Test Error {:.2f}%".format(_error(test_data, test_target))
    elif not args.epochs is None:
        net.train(train_data, train_target, n_epochs=args.epochs)      
        
    print "saving"
    net.save_data()
    
    if args.results:
        print "Train Error: {:.2f}%".format(_error(train_data, train_target))
        print "Test Error: {:.2f}%".format(_error(test_data, test_target))
    
    print "\nend " + str(datetime.now()) + "\n"
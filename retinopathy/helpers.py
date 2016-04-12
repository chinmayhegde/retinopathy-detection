import csv
import os
import random

import svm_classifier


def get_classifier_filename(classifier_name):
    dir_name = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_name, 'classifiers', classifier_name + '.pkl')


def get_image_split(csv_filename, image_folder):
    # Get image names and classifications
    image_names = []
    image_classes = {}
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # skip the first row
            if row[1] == 'level':
                continue
            # uncomment this to get only two categories
            # if int(row[1]) in [1, 2, 3]:
            #     continue
            image_name = image_folder + '/' + row[0] + '.jpeg'
            image_names.append(image_name)
            image_classes[image_name] = int(row[1])
    random.shuffle(image_names)

    train_ratio = 0.75
    train_num = int(train_ratio * len(image_names))

    train_filenames = image_names[:train_num]
    test_filenames = image_names[train_num:]
    return train_filenames, test_filenames, image_classes


def get_image_split2(image_folder):
    image_names = []
    image_classes = {}
    for image_class in range(5):
        class_image_names = os.listdir(image_folder + '/' + str(image_class))
        class_image_names = [image_folder + '/' + str(image_class) + '/' + name
                             for name in class_image_names]
        image_names.extend(class_image_names)
        for image_name in class_image_names:
            image_classes[image_name] = image_class
    random.shuffle(image_names)

    train_ratio = 0.8
    train_num = int(train_ratio * len(image_names))

    train_filenames = image_names[:train_num]
    test_filenames = image_names[train_num:]
    return train_filenames, test_filenames, image_classes


def get_classifier(classifier_name):
    if classifier_name == 'svm':
        return svm_classifier.SVMBatchClassifier([0, 1, 2, 3, 4])
    else:
        raise ValueError('invalid classifier: ' + classifier_name)


def get_fitted_classifier(classifier_name):
    if classifier_name == 'svm':
        return svm_classifier.SVMBatchClassifier.load()
    else:
        raise ValueError('invalid classifier: ' + classifier_name)


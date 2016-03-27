import cv2
import numpy
import csv
import random
import sys

from svm_classifier import SVMBatchClassifier


def get_batch(classifier, image_names, idx, batch_size):
    images = {}
    for image_name in image_names[idx:idx + batch_size]:
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image, ok = classifier.preprocess_image(image)
        if not ok:
            print 'preprocess failed, skipping image'
            continue
        images[image_name] = image
    return images


def run(classifier, csv_filename, image_folder):
    # Get image names and classifications
    names = []
    name_to_class = {}
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
            names.append(image_name)
            name_to_class[image_name] = int(row[1])
    random.shuffle(names)

    batch_size = 200
    # TODO check this
    train_ratio = 0.75
    train_num = int(train_ratio * len(names))
    for idx in range(0, train_num, batch_size):
        this_batch_size = min(train_num - idx, batch_size)
        image_map = get_batch(classifier, names, idx, this_batch_size)
        images = []
        image_classes = []
        for image_name, image in image_map.iteritems():
            images.append(image)
            image_classes.append(name_to_class[image_name])
        # print 'train:', idx

        train_labels = numpy.array(image_classes)
        train_data = numpy.array(images)
        classifier.train_batch(train_labels, train_data)

    correct = 0
    total = 0
    for idx in range(train_num, len(names), batch_size):
        this_batch_size = min(len(names) - idx, batch_size)
        image_map = get_batch(classifier, names, idx, this_batch_size)
        images = []
        image_classes = []
        for image_name, image in image_map.iteritems():
            images.append(image)
            image_classes.append(name_to_class[image_name])
        # print 'test:', idx

        test_labels = numpy.array(image_classes)
        test_data = numpy.array(images)

        result = classifier.predict(test_data)
        total += result.size
        for i in range(result.size):
            if result[i] == test_labels[i]:
                correct += 1

    print 'svm:', correct, 1.0 * correct / total


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: python main.py <classifier> <csv file> <image folder>'
        sys.exit(1)

    classifier = None
    # Add more casses for other classifiers
    if sys.argv[1] == 'svm':
        classifier = SVMBatchClassifier([0, 1, 2, 3, 4])
    else:
        print 'Invalid classifier:', sys.argv[1]
        sys.exit(1)

    for _ in range(10):
        run(classifier, sys.argv[2], sys.argv[3])


import cv2
import numpy
import csv
from sklearn import linear_model
import random
import sys


def get_batch(image_names, idx, batch_size, hog):
    images = {}
    for image_name in image_names[idx:idx + batch_size]:
        # median image size in (2592, 3888)
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (778, 518))
        # image = image.flatten().astype(numpy.float32)
        image = hog.compute(image)
        image = image.flatten()
        images[image_name] = image
    return images

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python main.py <csv file path> <image folder path>'
        sys.exit(1)

    # Get image names and classifications
    names = []
    name_to_class = {}
    with open(sys.argv[1], 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # skip the first row
            if row[1] == 'level':
                continue
            # if int(row[1]) in [1, 2, 3]:
            #     continue
            image_name = sys.argv[2] + '/' + row[0] + '.jpeg'
            names.append(image_name)
            name_to_class[image_name] = int(row[1])
    random.shuffle(names)

    # TODO: we should experiment with these
    win_size = (16, 16)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                            num_bins)

    svm_classifier = linear_model.SGDClassifier()
    batch_size = 200
    # TODO check this
    train_ratio = 0.75
    train_num = int(train_ratio * len(names))
    for idx in range(0, train_num, batch_size):
        this_batch_size = min(train_num - idx, batch_size)
        image_map = get_batch(names, idx, this_batch_size, hog)
        images = []
        image_classes = []
        for image_name, image in image_map.iteritems():
            images.append(image)
            image_classes.append(name_to_class[image_name])
        print 'train:', idx

        train_labels = numpy.array(image_classes)
        train_data = numpy.array(images)
        svm_classifier.partial_fit(train_data, train_labels,
                                   classes=[0, 1, 2, 3, 4])

    svm_correct = 0
    svm_total = 0
    for idx in range(train_num, len(names), batch_size):
        this_batch_size = min(len(names) - idx, batch_size)
        image_map = get_batch(names, idx, this_batch_size, hog)
        images = []
        image_classes = []
        for image_name, image in image_map.iteritems():
            images.append(image)
            image_classes.append(name_to_class[image_name])
        print 'test:', idx

        test_labels = numpy.array(image_classes)
        test_data = numpy.array(images)

        svm_result = svm_classifier.predict(test_data)
        svm_total += svm_result.size
        for i in range(svm_result.size):
            if svm_result[i] == test_labels[i]:
                svm_correct += 1

    print 'svm:', svm_correct, 1.0 * svm_correct / svm_total


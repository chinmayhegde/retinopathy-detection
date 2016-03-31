import cv2
import numpy
import pickle
from sklearn import linear_model

from classifier import BatchClassifier
import helpers
# from cnn import crop


class SVMBatchClassifier(BatchClassifier):

    def __init__(self, classes, batch_size=200):
        self.classes = classes
        self.batch_size = batch_size
        # win_size, block_size, block_stride, cell_size, num_bins
        self.hog = self._create_hog()
        self.classifier = linear_model.SGDClassifier()

    def _create_hog(self):
        return cv2.HOGDescriptor((16, 16), (16, 16), (8, 8), (8, 8), 9)

    def _preprocess_image(self, image):
        # median image size is (2592, 3888)
        image = cv2.resize(image, (778, 518))
        ok = True
        # image, ok = crop.crop_img(image)
        if not ok:
            return None, ok
        image = self.hog.compute(image)
        return image.flatten(), True

    def _get_batch(self, image_names):
        images = {}
        for image_name in image_names:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image, ok = self._preprocess_image(image)
            if not ok:
                print 'preprocess failed, skipping image'
                continue
            images[image_name] = image
        return images

    def train(self, image_names, image_classes):
        for idx in range(0, len(image_names), self.batch_size):
            image_map = self._get_batch(image_names[idx:idx + self.batch_size])

            images = []
            classes = []
            for image_name, image in image_map.iteritems():
                images.append(image)
                classes.append(image_classes[image_name])

            train_data = numpy.array(images)
            train_labels = numpy.array(classes)
            self.classifier.partial_fit(train_data, train_labels,
                                        classes=self.classes)

    def classify_test(self, image_names, image_classes):
        correct = 0
        total = 0
        for idx in range(0, len(image_names), self.batch_size):
            image_map = self._get_batch(image_names[idx:idx + self.batch_size])

            images = []
            classes = []
            for image_name, image in image_map.iteritems():
                images.append(image)
                classes.append(image_classes[image_name])

            test_data = numpy.array(images)
            test_labels = numpy.array(classes)

            result = self.classifier.predict(test_data)
            total += result.size
            for i in range(result.size):
                if result[i] == test_labels[i]:
                    correct += 1

        return correct, total

    def classify_single(self, image):
        data, ok = self._preprocess_image(image)
        if not ok:
            return -1
        return self.classifier.predict(data)

    def save(self):
        del self.hog

        with open(helpers.get_classifier_filename('svm'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load():
        with open(helpers.get_classifier_filename('svm'), 'rb') as f:
            classifier = pickle.load(f)
            classifier.hog = classifier._create_hog()


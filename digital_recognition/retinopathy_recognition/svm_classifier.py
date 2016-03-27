import cv2
from sklearn import linear_model

from classifier import BatchClassifier


class SVMBatchClassifier(BatchClassifier):

    def __init__(self, classes):
        self.classes = classes
        # win_size, block_size, block_stride, cell_size, num_bins
        self.hog = cv2.HOGDescriptor((16, 16), (16, 16), (8, 8), (8, 8), 9)
        self.classifier = linear_model.SGDClassifier()

    def preprocess_image(self, image):
        # median image size is (2592, 3888)
        image = cv2.resize(image, (778, 518))
        image = self.hog.compute(image)
        return image.flatten()

    def train_batch(self, train_labels, train_data):
        self.classifier.partial_fit(train_data, train_labels,
                                    classes=self.classes)

    def predict(self, test_data):
        return self.classifier.predict(test_data)


import cv2
import numpy

from classifier import Classifier


class KNNClassifier(Classifier):

    def classify(self, train_data, test_data, expected):
        knn = cv2.KNearest()
        knn.train(train_data, expected)

        _, result, _, _ = knn.find_nearest(test_data, k=5)

        # Find accuracy of predictions
        correct = 0
        for i in range(result.size):
            if result[i] == expected[i]:
                correct += 1

        return correct, 1.0 * correct / result.size

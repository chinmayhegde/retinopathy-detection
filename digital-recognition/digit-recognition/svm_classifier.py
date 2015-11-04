import cv2
import numpy

from classifier import Classifier


class SVMClassifier(Classifier):
    def classify(self, train_data, test_data, expected):
        svm_params = {
            'kernel_type': cv2.SVM_LINEAR,
            'svm_type': cv2.SVM_C_SVC,
            'C': 2.67,
            'gamma': 5.383
        }

        svm = cv2.SVM()
        svm.train(train_data, expected, params=svm_params)

        result = svm.predict_all(test_data)

        correct = 0
        for i in range(result.size):
            if result[i] == expected[i]:
                correct += 1

        return correct, 1.0 * correct / result.size

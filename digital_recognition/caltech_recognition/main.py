import os
import cv2
import numpy
import random


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    categories_path = os.path.join(dir_path, '101_ObjectCategories')
    categories = os.listdir(categories_path)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    winSize = (16,16)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    total_count = 0
    knn_count = 0
    svm_count = 0

    for _ in range(200):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for i in random.sample(range(len(categories)), 4): # pick four random categories
            category = categories[i]
            image_path = os.path.join(categories_path, category)
            image_list = os.listdir(image_path)
            image_count = 0

            for image_name in image_list:
                if image_count == 40:
                    break

                image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (300, 200))

                image = hog.compute(image)

                # first 20 go in testing
                if image_count < 20:
                    test_data.append(image)
                    test_labels.append([i])
                # second 20 go in training
                elif image_count < 40:
                    train_data.append(image)
                    train_labels.append([i])
                image_count += 1

        train_data = numpy.array(train_data)
        train_labels = numpy.array(train_labels)
        test_data = numpy.array(test_data)
        test_labels = numpy.array(test_labels)

        # KNN
        knn = cv2.KNearest()
        knn.train(train_data, train_labels)

        _, result, _, _ = knn.find_nearest(test_data, k=3)

        # Find accuracy of predictions
        correct = 0
        for i in range(result.size):
            if result[i] == test_labels[i]:
                correct += 1

        knn_count += correct
        # print 'knn:', correct, 1.0 * correct / result.size

        # SVM
        svm_params = {
            'kernel_type': cv2.SVM_LINEAR,
            'svm_type': cv2.SVM_C_SVC,
        }

        svm = cv2.SVM()
        svm.train(train_data, train_labels, params=svm_params)

        result = svm.predict_all(test_data)

        correct = 0
        for i in range(result.size):
            if result[i] == test_labels[i]:
                correct += 1

        svm_count += correct
        total_count += result.size
        # print 'svm:', correct, 1.0 * correct / result.size

    print 'knn:', knn_count, 1.0 * knn_count / total_count
    print 'svm:', svm_count, 1.0 * svm_count / total_count

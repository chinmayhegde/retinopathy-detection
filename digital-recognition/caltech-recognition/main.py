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
    hog = cv2.HOGDescriptor()

    for i in random.sample(range(len(categories)), 4): # pick four random categories
        category = categories[i]
        image_path = os.path.join(categories_path, category)
        image_list = os.listdir(image_path)
        image_count = 0

        for image_name in image_list:
            image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 200))
            image = hog.compute(image)
            flat = image.flatten().astype(numpy.float32)
            # first 20 go in testing
            if image_count < 20:
                test_data.append(flat)
                test_labels.append([i])
            # second 20 go in training
            elif image_count < 40:
                train_data.append(flat)
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

    print correct, 1.0 * correct / result.size

    # SVM
    svm_params = {
        'kernel_type': cv2.SVM_LINEAR,
        'svm_type': cv2.SVM_C_SVC,
        'C': 2.67,
        'gamma': 5.383
    }

    svm = cv2.SVM()
    svm.train(train_data, train_labels, params=svm_params)

    result = svm.predict_all(test_data)

    correct = 0
    for i in range(result.size):
        if result[i] == test_labels[i]:
            correct += 1

    print correct, 1.0 * correct / result.size

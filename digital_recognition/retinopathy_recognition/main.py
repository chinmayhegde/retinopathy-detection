import cv2
import numpy
import csv


def brightness_normalize():
    pass

if __name__ == '__main__':
    # Get image names and classifications
    names = {i: [] for i in range(5)}
    with open('subsetcsv.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            names[int(row[1])].append('train/' + row[0] + '.jpeg')

    # TODO: we should experiment with these
    win_size = (16, 16)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                            num_bins)

    # Load all images into memory for now
    images = {i: [] for i in range(5)}
    count = 0
    for classification, image_names in names.iteritems():
        for image_name in image_names:
            count += 1
            print 'loading:', count
            # median image size in (2592, 3888)
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (518, 778))
            image = image.flatten().astype(numpy.float32)
            # image = hog.compute(image)
            images[classification].append(image)

    # Partition images into test and train sets
    train_ratio = 0.75
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
    test_data = numpy.array(test_data)

    # Classify, move these to other classes
    # KNN
    knn = cv2.KNearest()
    knn.train(train_data, train_labels)
    _, knn_result, _, _ = knn.find_nearest(test_data, k=3)

    knn_correct = 0
    for i in range(knn_result.size):
        if knn_result[i] == test_labels[i]:
            knn_correct += 1
    print 'knn:', knn_correct, 1.0 * knn_correct / knn_result.size

    # SVM
    svm_params = {
        'kernel_type': cv2.SVM_LINEAR,
        'svm_type': cv2.SVM_C_SVC,
    }

    svm = cv2.SVM()
    svm.train(train_data, train_labels, params=svm_params)
    svm_result = svm.predict_all(test_data)

    svm_correct = 0
    for i in range(svm_result.size):
        if svm_result[i] == test_labels[i]:
            svm_correct += 1
    print 'svm:', svm_correct, 1.0 * svm_correct / svm_result.size


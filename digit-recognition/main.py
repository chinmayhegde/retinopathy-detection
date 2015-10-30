
import utils
from knn_classifier import KNNClassifier
from svm_classifier import SVMClassifier


if __name__ == '__main__':
    train_data, test_data, expected = utils.get_default_image_transformed()

    knn = KNNClassifier()
    count, percent = knn.classify(train_data, test_data, expected)
    print 'knn correct count', count
    print 'knn correct percentage', percent

    svm = SVMClassifier()
    count, percent = svm.classify(train_data, test_data, expected)
    print 'svm correct count', count
    print 'svm correct percentage', percent


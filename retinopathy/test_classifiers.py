import sys

import helpers


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python main.py <classifiers> <image folder>'
        sys.exit(1)

    classifiers = sys.argv[1].split(',')
    image_folder = sys.argv[2]

    for _ in range(10):
        train_filenames, test_filenames, image_classes = \
            helpers.get_image_split2(image_folder)

        for classifier_name in classifiers:
            classifier = helpers.get_classifier(classifier_name)

            classifier.train(train_filenames, image_classes)
            correct, total = classifier.classify_test(test_filenames,
                                                      image_classes)

            print 'result:', correct, 1.0 * correct / total


import sys

import helpers


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: python main.py <classifiers> <csv file> <image folder>'
        sys.exit(1)

    classifiers = sys.argv[1].split(',')
    csv_filename = sys.argv[2]
    image_folder = sys.argv[3]

    for _ in range(10):
        train_filenames, test_filenames, image_classes = \
            helpers.get_image_split(csv_filename, image_folder)

        for classifier_name in classifiers:
            classifier = helpers.get_classifier(classifier_name)

            classifier.train(train_filenames, image_classes)
            correct, total = classifier.classify_test(test_filenames,
                                                      image_classes)

            print 'result:', correct, 1.0 * correct / total


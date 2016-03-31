

class BatchClassifier:
    # Implement methods of this class to use it in main.py

    def train(self, filenames, image_classes):
        '''
        Train the classifier.

        :param filenames: the list of filenames to be used for training
        :type filenames: [str]
        :param image_classes: a mapping of filename to class
        :type image_classes: {str: int}
        '''
        raise NotImplemented

    def classify_test(self, filenames, image_classes):
        '''
        Test the accuracy of the classifier with known data.

        :param filenames: the list of filenames to be used for training
        :type filenames: [str]
        :param image_classes: a mapping of filename to class
        :type image_classes: {str: int}
        '''
        raise NotImplemented

    def classify_single(self, data):
        '''
        Classify a given image.

        :param data: the image data to classify
        :type data: cv2 image representation
        '''
        raise NotImplemented

    def save(self):
        '''
        Save the classifier to 'classifier/<name>.pkl'
        '''
        raise NotImplemented


from flask import Blueprint, render_template, request
import cv2
import numpy
import svm_classifier


retinopathy_view = Blueprint('retinopathy_view', __name__)


@retinopathy_view.route('/', methods=['GET', 'POST'])
def display_home_page():
    # TODO
    if request.method == 'POST':
        f = request.files['image']
        image_bytes = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
        classifier = svm_classifier.SVMBatchClassifier.load()
        image_class = classifier.classify_single(image)

        return render_template('home.html', image_class=image_class)
    return render_template('home.html')


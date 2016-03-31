from flask import Blueprint, render_template, request
import cv2
import numpy
# import svm_classifier
import helpers


retinopathy_view = Blueprint('retinopathy_view', __name__)


@retinopathy_view.route('/', methods=['GET', 'POST'])
def display_home_page():
    # TODO
    if request.method == 'POST':
        classifier_name = request.form.get('classifier')
        classifier = helpers.get_fitted_classifier(classifier_name)

        f = request.files['image']
        image_bytes = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
        image_class = classifier.classify_single(image)

        return render_template('home.html', image_class=image_class)
    return render_template('home.html')


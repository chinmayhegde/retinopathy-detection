from flask import Blueprint, render_template, request
import cv2
import numpy


user_view = Blueprint('user_view', __name__)


@user_view.route('/', methods=['GET', 'POST'])
def display_home_page():
    # TODO
    if request.method == 'POST':
        f = request.files['image']
        image_bytes = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
        print image.__class__

        return render_template('home.html', image_class='TODO')
    return render_template('home.html')


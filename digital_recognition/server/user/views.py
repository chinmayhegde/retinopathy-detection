from flask import Blueprint, render_template, request
import json


user_view = Blueprint('user_view', __name__)


@user_view.route('/', methods=['GET', 'POST'])
def display_home_page():
    if request.method == 'POST':
        image = request.files['image']
        # TODO remove this when we do actual stuff
        print image.__class__

        return render_template('home.html', image_class='TODO')
    return render_template('home.html')


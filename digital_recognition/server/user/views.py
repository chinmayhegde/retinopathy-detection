from flask import Blueprint
import json


user_view = Blueprint('user_view', __name__)

@user_view.route('/')
def display_home_page():
    # Can return HTML page with 'render_template()'
    return 'Hello, world!'


@user_view.route('/classify', methods=['POST'])
def classify_image():
    # TODO
    return json.dumps(-1)


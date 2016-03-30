from flask import Flask
from user.views import user_view


def create_app():
    app = Flask(__name__)
    app.register_blueprint(user_view)
    return app


if __name__ == '__main__':
    app = create_app()
    app.run()


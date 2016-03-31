from flask import Flask
import retinopathy.views as retinopathy


def create_app():
    app = Flask(__name__)
    app.register_blueprint(retinopathy.retinopathy_view)
    app.debug = True
    return app


if __name__ == '__main__':
    app = create_app()
    app.run()


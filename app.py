import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask

from blueprint import api_bp
from container import Container

LOG_FILE_PATH = 'logs'
LOG_FILE_NAME = 'app.log'

app = Flask(__name__)

def setup_logging():
    os.makedirs(LOG_FILE_PATH, exist_ok=True)

    for handler in list(app.logger.handlers):
        app.logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

    file_handler = RotatingFileHandler(os.path.join(LOG_FILE_PATH, LOG_FILE_NAME), maxBytes=1024 * 1024 * 10, backupCount=10)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)


def create_app():
    app_container = Container()

    args_env = os.environ.get('FLASK_ENV', 'dev')  # This is used for 'env'
    args_debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    setup_logging()
    app.logger.info(f"Starting app initialization with env={args_env}, debug={args_debug}")

    app.container = app_container

    app_container.config.app_args.from_dict({'env': args_env})

    app.config['DEBUG'] = args_debug
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_dev')

    app.register_blueprint(api_bp)
    app.logger.info("API Blueprint registered.")

    app.logger.info(f'App initialized successfully: env={args_env}, debug={args_debug}')

    return app


if __name__ == '__main__':
    initialized_app = create_app()
    initialized_app.run(host='0.0.0.0', port=8000)
    logging.getLogger().info("Flask development server stopped.")

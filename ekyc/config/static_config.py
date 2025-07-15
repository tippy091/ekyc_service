import configparser
import os


class StaticConfig:
    asset_config = None
    app_args = None

    def __init__(self, app_args):
        self.app_args = app_args
        self.read_config()

    def read_config(self):
        """Reads configuration from an INI file."""
        self.asset_config = configparser.ConfigParser(os.environ)

        config_file = os.path.join(os.path.dirname(__file__), "env", f"{self.app_args['env']}.ini")

        self.asset_config.read(config_file)
        pass

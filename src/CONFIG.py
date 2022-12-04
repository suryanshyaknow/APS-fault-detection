import yaml
import os
from src.logger import lg
from dataclasses import dataclass


@dataclass
class Config:
    """Helps to access the secret and protected params that cannot be exposed beyond the scope of the project.
    Moreover, provides us with a method to read the global params from the provided path of the configuration
    file.
    """
    mongodb_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key:str = os.getenv("AWS_SECERET_ACCESS_KEY")

    def read_params(self, config_file_path):
        """Read and return the dictionary containing all the paths and params from the parameterized
        configuration file's path.

        Args:
            file_path (string): Path of the configuration file.

        Returns:
            dict: Contains all the params and paths in the form of key:val pair.
        """
        try:
            with open(config_file_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            lg.exception(e)


if __name__ == "__main__":
    Config()
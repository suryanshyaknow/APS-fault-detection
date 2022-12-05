import yaml
import os
from dotenv import load_dotenv
from src.logger import lg
from dataclasses import dataclass


@dataclass
class Config:
    """Helps to access the secret and protected params declared inside the .env that cannot be exposed beyond 
    the scope of the project.
    """
    load_dotenv()
    mongodb_url: str = os.getenv("MONGO_DB_URL")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECERET_ACCESS_KEY")


def read_params(config_file_path):
    """Retrieve all the params or paths declared inside the desired config file and returns the dictionary
    containing all of them.

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
import yaml
import argparse
from src.logger import lg

def read_params(file_path):
    """Read and return the dictionary containing all the paths and params from the parameterized
    configuration file's path.

    Args:
        file_path (string): Path of the configuration file.

    Returns:
        dict: Contains all the params and paths in the form of key:val pair.
    """
    try:
        with open(file_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        lg.exception(e)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", default="params.yaml")
    parse_args = args_parser.parse_args()
    read_params(file_path=parse_args.config)

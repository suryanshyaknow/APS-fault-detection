import yaml
from src.logger import lg
import os


def write_yaml_file(file_path:str, data:dict):
    """Dumps the desired data into `yaml` file at the said location.

    Args:
        file_path (str): Location where yaml file is to be created.
        data (dict): Data that is to be dumped into yaml file.
    """ 
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f)
        ...
    except Exception as e:
        lg.exception(e)
    

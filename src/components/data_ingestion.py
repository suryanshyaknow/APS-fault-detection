from src.utils.db_operations import dBOperations
from src.logger import lg
import pandas as pd
import numpy as np
import os
import argparse
from dataclasses import dataclass
from src.CONFIG import read_params
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestion:
    lg.info(f'Entered the "{os.path.basename(__file__)}.DataIngestion" class')
    config_file_path: str
    feature_store_path: str = None
    test_size: float = None
    test_path: str = None
    training_path: str = None
    random_state: int = None

    def fetch_params(self):
        """This method fetches the desired params from the configuration file.
        """
        try:
            lg.info("fetching the params from the config file..")
            config_params = read_params(self.config_file_path)
            self.feature_store_path = config_params["data_ingestion"]["feature_store_path"]
            self.test_size = config_params["data_ingestion"]["test_size"]
            self.training_path = config_params["data_ingestion"]["training_data_path"]
            self.test_path = config_params["data_ingestion"]["test_data_path"]
            self.random_state = config_params["data_ingestion"]["random_state"]
            ...
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("params fetched successfully!")

    def initiate(self):
        try:
            # Readying the "sensors" dataframe
            self.fetch_params()
            lg.info('Exporting the "sensors" data as pandas dataframe..')
            df: pd.DataFrame = dBOperations(
                self.config_file_path).getDataAsDataFrame()
            lg.info("dataframe fetched!")
            # Replacing "na" values with np.NaN in the dataframe
            df.replace(to_replace="na", value=np.NaN, inplace=True)
            # Making sure the dir where dataframe is to be stored does exist
            df_dir = os.path.dirname(self.feature_store_path)
            os.makedirs(df_dir, exist_ok=True)
            # Saving the dataframe to the feature_store_path
            df.to_csv(path_or_buf=self.feature_store_path, index=None)
            lg.info('"sensors" dataframe exported successfully!')

            # TRAIN-TEST SPLIT
            lg.info('Splitting the data into training and test subsets..')
            training_set, test_set = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state)
            lg.info("data split into test and training subsets successfully!    ")
            # Making sure the test and training dirs do exist
            test_dir = os.path.dirname(self.test_path)
            training_dir = os.path.dirname(self.training_path)
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(training_dir, exist_ok=True)
            # Saving the test and train set to their respective dirs
            lg.info("Saving the test and training subsets to their respective dirs..")
            test_set.to_csv(path_or_buf=self.test_path, index=None)
            training_set.to_csv(path_or_buf=self.training_path, index=None)
            lg.info("test and training subsets saved succesfully!")
            ...
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("DATA INGESTION completed!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", default="./params.yaml")
    parse_args = args_parser.parse_args()
    DataIngestion(config_file_path=parse_args.config).initiate()

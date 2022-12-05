import pymongo
import pandas as pd
import argparse
from src.CONFIG import Config, read_params
from src.logger import lg
import os
from dataclasses import dataclass


@dataclass
class dBOperations:
    lg.info(f'Entered the "{os.path.basename(__file__)}.dBOperations" class')
    config_file_path: str
    client = None
    database = None
    collection = None
    connection_url: str = None
    db_name: str = None
    collection_name: str = None

    def fetch_params(self):
        """This method fetches the desired params from the configuration file.
        """
        try:
            lg.info("fetching the params from .env and config file..")
            # Reading the params from .env
            config = Config()
            self.connection_url = config.mongodb_url

            # Reading the paramsn from config file
            config_params = read_params(config_file_path=self.config_file_path)
            self.data_path = config_params["data_source"]["raw_data_path"]
            self.db_name = config_params["data_source"]["MongoDB_database_name"]
            self.collection_name = config_params["data_source"]["MongoDB_collection_name"]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("params fetched successfully!")

    def establishConnectionToMongoDB(self):
        """This method establishes the connection to the MongoDB Cluster.
        """
        try:
            self.fetch_params()
            lg.info("Establishing the connection to MongoDB..")
            self.client = pymongo.MongoClient(self.connection_url)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("connection established successfully!")

    def selectDB(self):
        """This method chooses the desired dB from the MongoDb Cluster.
        """
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.db_name]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(f'"{self.db_name}" database chosen succesfully!')

    def selectCollection(self):
        """This method chooses the collection from the selected database of the MongoDB Cluster.
        """
        try:
            self.selectDB()
            lg.info("searching for the collection..")
            self.collection = self.database[self.collection_name]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'"{self.collection_name}" collection in the database "{self.db_name}" selected successfully!')

    def getDataAsDataFrame(self) -> pd.DataFrame:
        """This method returns the sensors-streaming-data as dataframe.

        Returns:
            pandas.DataFrame: Data from the given collection of the MongoDB database in form of pandas dataframe.
        """
        try:
            self.selectCollection()
            lg.info(
                f'reading the data from collection "{self.collection_name}" of database "{self.db_name}"..')
            df = pd.DataFrame(list(self.collection.find()))
            lg.info("data fetched into the dataframe!")
            df.drop(columns=["_id"], inplace=True)
            lg.info(f"Shape of the data: {df.shape}")
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("returning the database..")
            return df


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", default="./params.yaml")
    parse_args = args_parser.parse_args()
    dBOperations(config_file_path=parse_args.config)

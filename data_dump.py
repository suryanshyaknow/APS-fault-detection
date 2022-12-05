import pymongo
import pandas as pd
import argparse
from src.CONFIG import Config, read_params
from src.logger import lg
import json
import os
from dataclasses import dataclass


@dataclass
class DumpDataToMongoDB:
    lg.info(
        f'Entered the "{os.path.basename(__file__)}.DumpDataToMongoDB" class')
    config_file_path: str
    data_path: str = None
    client: str = None
    database_name: str = None
    database: str = None
    collection_name: str = None
    collection: str = None

    def fetch_params(self):
        try:
            lg.info("fetching the params from .env and the configuration file..")
            # fecthing params from .env
            config = Config()
            self.connection_url = config.mongodb_url
            # fetching the relevants from the configuration file
            config_params = read_params(config_file_path=self.config_file_path)
            self.data_path = config_params["data_source"]["raw_data_path"]
            self.db_name = config_params["data_source"]["MongoDB_database_name"]
            self.collection_name = config_params["data_source"]["MongoDB_collection_name"]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("params fetched successfully..")

    def establishConnectionToMongoDB(self):
        try:
            self.fetch_params()
            lg.info("Establishing the connection to MongoDB..")
            self.client = pymongo.MongoClient(self.connection_url)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("connection established successfully!")

    def createOrSelectDB(self):
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.db_name]

        except Exception as e:
            lg.exception(e)

        else:
            lg.info(f'"{self.db_name}" database chosen succesfully!')

    def createCollection(self):
        try:
            self.createOrSelectDB()
            lg.info("creating the collection..")
            self.collection = self.database[self.collection_name]

        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'"{self.collection_name}" collection in the database "{self.db_name}" created successfully!')

    def dumpData(self):
        try:
            self.createCollection()
            lg.info("fetching data from the raw data path..")
            df = pd.read_csv(self.data_path)
            lg.info(f"shape of the data: {df.shape}")
            lg.info("data fetched..")
            # Converting the .csv into "dumpable into MongoDB" format --json
            df.reset_index(drop=True, inplace=True)
            all_records = list(json.loads(df.T.to_json()).values())

            # Inserting into the Collection
            self.collection.insert_many(all_records)

        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'successfully dumped all data from "{self.data_path}" into database "{self.db_name}" in MongoDB!')


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", default="./params.yaml")
    parse_args = args_parser.parse_args()

    # Creating an object of class `DumpDataToMongoDB` to dump data to MongoDB
    dump_data = DumpDataToMongoDB(config_file_path=parse_args.config)
    # dumping the data
    dump_data.dumpData()

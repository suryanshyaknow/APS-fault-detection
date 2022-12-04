import pymongo
import pandas as pd
import argparse
from src.CONFIG import Config
from src.logger import lg
import json


class DumpDataToMongoDB:
    def __init__(self, config_file_path):
        try:
            lg.info(
                f"Entered the __init__ method of the {self.__class__.__name__}")
            lg.info("fetching the params from the configuration file..")
            
            config = Config()
            self.connection_url = config.mongodb_url
            # self.connection_url = "mongodb+srv://suryanshyaknow:streamingdata@sensors-streaming-data.kkw0g1z.mongodb.net/test"

            global_params = config.read_params(config_file_path=config_file_path)
            # fetching the relevants from the configuration file
            self.data_path = global_params["data_source"]["raw_data_path"]
            self.db_name = global_params["data_source"]["MongoDB_database_name"]
            self.collection_name = global_params["data_source"]["MongoDB_collection_name"]
            lg.info("params fetched successfully!")

            self.client = None
            self.database = None
            self.collection = None

        except Exception as e:
            lg.exception(e)

    def establishConnectionToMongoDB(self):
        try:
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
            lg.info("fetching the data from the raw data path..")
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

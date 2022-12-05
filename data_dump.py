import pymongo
import pandas as pd
from src.CONFIG import Config
from src.entities.config import DataSourceConfig
from src.logger import lg
import json
import os
from dataclasses import dataclass


@dataclass
class DumpDataToMongoDB:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DumpDataToMongoDB" class')
    data_source_config: DataSourceConfig = DataSourceConfig()
    client: str = None
    database: str = None
    collection: str = None

    def establishConnectionToMongoDB(self):
        try:
            # Connection URL
            connection_url = Config().mongodb_url
            lg.info("Establishing connection to MongoDB..")
            self.client = pymongo.MongoClient(connection_url)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("connection established successfully!")

    def createOrSelectDB(self):
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.data_source_config.database_name]

        except Exception as e:
            lg.exception(e)

        else:
            lg.info(
                f'"{self.data_source_config.database_name}" database chosen succesfully!')

    def createCollection(self):
        try:
            self.createOrSelectDB()
            lg.info("creating the collection..")
            self.collection = self.database[self.data_source_config.collection_name]

        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'"{self.data_source_config.collection_name}" collection in the database "{self.data_source_config.database_name}" created successfully!')

    def dumpData(self):
        try:
            self.createCollection()
            lg.info("fetching data from the raw data path..")
            df = pd.read_csv(self.data_source_config.raw_data_path)
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
                f'successfully dumped all data from "{self.data_source_config.raw_data_path}" into database "{self.data_source_config.database_name}" in MongoDB!')


if __name__ == "__main__":
    # Creating an object of class `DumpDataToMongoDB` to dump data to MongoDB
    dump_data = DumpDataToMongoDB()
    # dumping the data
    dump_data.dumpData()

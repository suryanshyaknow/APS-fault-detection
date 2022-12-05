import pymongo
import pandas as pd
from src.CONFIG import Config
from src.entities.config import DataSourceConfig
from src.logger import lg
import os
from dataclasses import dataclass


@dataclass
class dBOperations:
    lg.info(f'Entered the "{os.path.basename(__file__)[:-3]}.dBOperations" class')
    data_source_config = DataSourceConfig()
    client = None
    database = None
    collection = None

    def establishConnectionToMongoDB(self):
        """This method establishes the connection to the MongoDB Cluster.
        """
        try:
            lg.info("Establishing the connection to MongoDB..")
            connection_url = Config().mongodb_url
            self.client = pymongo.MongoClient(connection_url)
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("connection established successfully!")

    def selectDB(self):
        """This method chooses the desired dB from the MongoDB Cluster.
        """
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.data_source_config.database_name]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(f'"{self.data_source_config.database_name}" database chosen succesfully!')

    def selectCollection(self):
        """This method chooses the collection from the selected database of the MongoDB Cluster.
        """
        try:
            self.selectDB()
            lg.info("searching for the collection..")
            self.collection = self.database[self.data_source_config.collection_name]
        except Exception as e:
            lg.exception(e)
        else:
            lg.info(
                f'"{self.data_source_config.collection_name}" collection in the database "{self.data_source_config.database_name}" selected successfully!')

    def getDataAsDataFrame(self) -> pd.DataFrame:
        """This method returns the sensors-streaming-data as dataframe.

        Returns:
            pandas.DataFrame: Data from the given collection of the MongoDB database in form of pandas dataframe.
        """
        try:
            self.selectCollection()
            lg.info(
                f'reading the data from collection "{self.data_source_config.collection_name}" of database "{self.data_source_config.database_name}"..')
            df = pd.DataFrame(list(self.collection.find()))
            lg.info("data readied as dataframe!")
            df.drop(columns=["_id"], inplace=True)
            lg.info(f"Shape of the data: {df.shape}")
        except Exception as e:
            lg.exception(e)
        else:
            lg.info("returning the database..")
            return df


if __name__ == "__main__":
    dBOperations()
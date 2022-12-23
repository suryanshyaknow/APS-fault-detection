import pymongo
import pandas as pd
from src.CONFIG import Config
from src.entities.config import DataSourceConfig
from src.logger import lg
import os
from dataclasses import dataclass


@dataclass
class dBOperations:
    """This class is exclusively for performing all MongoDB pertaining operations.
    
    Args:
        connection_string (str): Takes in the `client url` to establish connection to MongoDB.
        database_name (str): Database to which connection is to be established.
        collection_name (str): Desired collection name of the said database. 
    """
    lg.info(f'Entered the "{os.path.basename(__file__)[:-3]}.dBOperations" class')
    data_source_config = DataSourceConfig()
    client = None
    database = None
    collection = None

    def establishConnectionToMongoDB(self):
        """This method establishes the connection to the MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while establishing connection to MongoDB.
        """
        try:
            lg.info("Establishing the connection to MongoDB..")
            connection_url = Config().mongodb_url
            self.client = pymongo.MongoClient(connection_url)
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("connection established successfully!")

    def selectDB(self):
        """This method chooses the desired dB from the MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while selecting desired database from MongoDB.
        """
        try:
            self.establishConnectionToMongoDB()
            lg.info("searching for the database..")
            self.database = self.client[self.data_source_config.database_name]
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(f'"{self.data_source_config.database_name}" database chosen succesfully!')

    def selectCollection(self):
        """This method chooses the collection from the selected database of the MongoDB Cluster.

        Raises:
            e: Throws exception if any error pops up while selecting any desired collection in selected database of MongoDB.
        """
        try:
            self.selectDB()
            lg.info("searching for the collection..")
            self.collection = self.database[self.data_source_config.collection_name]
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(
                f'"{self.data_source_config.collection_name}" collection in the database "{self.data_source_config.database_name}" selected successfully!')

    def getDataAsDataFrame(self) -> pd.DataFrame:
        """This method returns the sensors-streaming-data as dataframe.

        Raises:
            e: Throws exception if any error pops up while loading data as dataframe from MongoDB's database.

        Returns:
            pandas.DataFrame: Data from the given collection of the MongoDB database in form of pandas dataframe.
        """
        try:
            self.selectCollection()
            lg.info(
                f'reading data from the collection "{self.data_source_config.collection_name}" of the database "{self.data_source_config.database_name}"..')
            df = pd.DataFrame(list(self.collection.find()))
            lg.info("data readied as the dataframe!")
            df.drop(columns=["_id"], inplace=True)
            lg.info(f"Shape of the data: {df.shape}")
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("returning the database..")
            return df


if __name__ == "__main__":
    dBOperations()
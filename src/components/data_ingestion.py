from src.utils.db_operations import dBOperations
from src.logger import lg
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.entities.config import DataIngestionConfig
from src.entities.artifact import DataIngestionArtifact
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestion:
    """Shall be used for obtaining and importing data from the desired dB in a form of feature store file.
    Data is also being split into train and test subsets herein this stage only.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataIngestion" class')

    data_ingestion_config = DataIngestionConfig()

    def initiate(self) -> DataIngestionArtifact:
        """Initiates the Data Ingestion stage of the training piepeline.

        Raises:
            e: Raises relevant exception should any sort of error pops up while ingestion of data.

        Returns:
            DataIngestionArtifact: Contains configurations of `feature-store file`, `training set` and `test set`. 
        """
        try:
            lg.info(f"\n{'='*27} DATA INGESTION {'='*40}")

            ################################# Readying the "sensors" dataframe #################################
            lg.info('Exporting the "sensors" data as pandas dataframe..')
            df: pd.DataFrame = dBOperations().getDataAsDataFrame()
            lg.info("dataframe fetched!")
            # Replacing "na" values with np.NaN in dataframe
            df.replace(to_replace="na", value=np.NaN, inplace=True)
            # Making sure the dir where dataframe is to be stored does exist
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            # Saving the dataframe to the feature_store_path
            df.to_csv(
                path_or_buf=self.data_ingestion_config.feature_store_file_path, index=None)
            lg.info('"sensors" dataframe exported successfully!')

            ###################################### TRAINING-TEST SPLIT #########################################
            lg.info('Splitting the data into training and test subsets..')
            training_set, test_set = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=self.data_ingestion_config.random_state)
            lg.info("data split into test and training subsets successfully!")
            # Making sure the test and training dirs do exist
            test_dir = os.path.dirname(
                self.data_ingestion_config.test_file_path)
            training_dir = os.path.dirname(
                self.data_ingestion_config.training_file_path)
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(training_dir, exist_ok=True)
            # Saving the test and train set to their respective dirs
            lg.info("Saving the test and training subsets to their respective dirs..")
            test_set.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path, index=None)
            training_set.to_csv(
                path_or_buf=self.data_ingestion_config.training_file_path, index=None)
            lg.info("test and training subsets saved succesfully!")
            
            #################################### Saving Artifacts Config #######################################
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file=self.data_ingestion_config.feature_store_file_path,
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            lg.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            lg.info("DATA INGESTION completed!")
            return data_ingestion_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e

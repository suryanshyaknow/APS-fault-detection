from src.logger import lg
from dataclasses import dataclass
import os
from src.entities.config import DataIngestionConfig, DataValidationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


@dataclass
class TrainingPipeline:
    lg.info("Training Pipeline begins now..")
    lg.info(f"Entered the {os.path.basename(__file__)[:-3]}.TrainingPipeline")
    data_ingestion_config: DataIngestionConfig
    data_validation_config: DataValidationConfig

    def begin(self):
        try:
            ######################### DATA INGESTION ####################################
            ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config)
            ingestion_artifact = ingestion.initiate()

            ######################### DATA VALIDATION ####################################
            validation = DataValidation(
                data_validation_config=self.data_validation_config, data_ingestion_artifact=ingestion_artifact)
            validation_artifact = validation.initiate()

            ...
        except Exception as e:
            lg.exception(e)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline(
        data_ingestion_config=DataIngestionConfig(), data_validation_config=DataValidationConfig())
    training_pipeline.begin()

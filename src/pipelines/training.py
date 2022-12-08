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

    def begin(self):
        try:
            ######################### DATA INGESTION ####################################
            ingestion = DataIngestion()
            ingestion_artifact = ingestion.initiate()

            ######################### DATA VALIDATION ####################################
            validation = DataValidation(
                data_ingestion_artifact=ingestion_artifact)
            validation_artifact = validation.initiate()

            ...
        except Exception as e:
            lg.exception(e)


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.begin()

from src.logger import lg
from dataclasses import dataclass
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pushing import ModelPushing


@dataclass
class TrainingPipeline:
    """Shall be used for triggering the Training pipeline."""
    lg.info("Training Pipeline begins now..")
    lg.info(f"Entered the {os.path.basename(__file__)[:-3]}.TrainingPipeline")

    def begin(self) -> None:
        """Commences the training pipeline starting from Data Ingestion component followed by Data Validation, Data Transformation,
        Model Training, Model Evaluation and at last, Model Pushing.

        Raises:
            e: Raises exception should any sort of error pops up during the training pipeline flow execution.
        """
        try:
            ######################### DATA INGESTION #######################################
            ingestion = DataIngestion()
            ingestion_artifact = ingestion.initiate()

            ######################### DATA VALIDATION ######################################
            validation = DataValidation(
                data_ingestion_artifact=ingestion_artifact)
            validation_artifact = validation.initiate()

            ######################### DATA TRANSFORMATION ##################################
            transformation = DataTransformation(
                data_ingestion_artifact=ingestion_artifact)
            transformation_artifact = transformation.initiate()

            ######################### MODEL TRAINING #######################################
            model_training = ModelTraining(
                data_transformation_artifact=transformation_artifact)
            model_training_artifact = model_training.initiate()

            ######################### MODEL EVALUATION #####################################
            model_evaluation = ModelEvaluation(
                data_ingestion_artifact=ingestion_artifact,
                data_transformation_artifact=transformation_artifact,
                model_training_artifact=model_training_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate()

            ######################### MODEL PUSHING ########################################
            model_pushing = ModelPushing(
                data_transformation_artifact=transformation_artifact,
                model_training_artifact=model_training_artifact
            )
            model_pushing_artifact = model_pushing.initiate()
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info("Training Pipeline ran with success!")


if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.begin()

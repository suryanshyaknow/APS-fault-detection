import os
import pandas as pd
import numpy as np
from src.CONFIG import ModelRegistryConfig
from src.entities import config, artifact
from src.utils.file_operations import BasicUtils
from sklearn.metrics import f1_score
from src.logger import lg
from dataclasses import dataclass


@dataclass
class ModelEvaluation:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelEvaluation" class')

    data_ingestion_artifact: artifact.DataIngestionArtifact
    data_transformation_artifact: artifact.DataTransformationArtifact
    model_training_artifact: artifact.ModelTrainingArtifact

    model_eval_config = config.ModelEvaluationConfig()
    model_registry_config = ModelRegistryConfig()
    target = config.BaseConfig().target

    def initiate(self):
        try:
            lg.info(f"\n{'='*27} MODEL EVALUATION {'='*40}")
            latest_dir = self.model_registry_config.get_latest_dir_path()

            ################### Compare the latest model to an old one if there's one #########################
            # If there's no old model, then the configure the current one
            if latest_dir is None:
                lg.info(
                    "Configuring the current model as there's no old model present to get compared to it..")
                model_eval_artifact = artifact.ModelEvaluationArtifact(
                    is_model_replaced=True, improved_accuracy=None)
                lg.info(f"Model Evaluation Artifact: {model_eval_artifact}")

            else:   
                ################ Load Model and Transformer already present there in Model Registry ################
                # Fetching their locations
                lg.info(
                    "fetching the locations of older model and transformer pipeline from the Model Registry..")
                model_path = self.model_registry_config.get_latest_model_path()
                transformer_pipeline_path = self.model_registry_config.get_latest_transformer_path()
                lg.info("locations fetched successfully!")
                # Loading them
                lg.info("Now, loading the older model and transformer pipeline..")
                older_model = BasicUtils.load_object(
                    model_path, obj_desc="older Model")
                older_transformer_pipeline = BasicUtils.load_object(
                    transformer_pipeline_path, obj_desc="older Transformer Pipeline")

                ########################### Load latest Model and Transformer ######################################
                lg.info("Now, loading the latest model and transformer pipeline..")
                latest_model = BasicUtils.load_object(
                    self.model_training_artifact.model_path, obj_desc="latest Model")
                latest_transformer_pipeline = BasicUtils.load_object(
                    self.data_transformation_artifact.transformer_path, obj_desc="latest Transformer Pipeline")

                ############################ Load the test dataframe ###############################################
                lg.info("loading the test dataset from the `data ingestion artifact`..")
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)


                ########################### Evaluating the older Model's performance ###############################
                # Pass the test dataset through the older Transformation Pipeline
                lg.info("Evaluating the performance of the `older model`..")
                test_transformed_array = older_transformer_pipeline.transform(
                    test_df)
                # Configure the transformed test features for making predicitions
                X_test_arr, y_true = test_transformed_array[:, :-1], test_transformed_array[:, -1]
                lg.info(
                    "Making predictions on the test dataset using the `older model`..")
                y_pred = older_model.predict(X_test_arr)
                lg.info("predictions made successfully!")
                lg.info(
                    "Evaluating the performance of the `older model` using \"f1-score\"..")
                older_model_score = f1_score(y_true, y_pred)
                lg.info(f"Older Model's performance: {older_model_score}")

                ########################## Evaluating the latest Model's performance ###############################
                lg.info("Evaluating the performance of the `older model`..")
                # Pass the test dataset through the latest Transformation Pipeline
                test_transformed_array = latest_transformer_pipeline.transform(
                    test_df)
                # Configure the transformed test features for making predictions
                X_test_arr, y_true = test_transformed_array[:, :-1], test_transformed_array[:, -1]
                lg.info(
                    "Making predictions on the test dataset using the `latest model`..")
                y_pred = latest_model.predict(X_test_arr)
                lg.info("predictions made successfully!")
                lg.info(
                    "Evaluating the performance of the `latest model` using \"f1-score\"..")
                latest_model_score = f1_score(y_true, y_pred)
                lg.info(f"Latest Model's performance: {latest_model_score}")

                ################## Comparison between the two versions to keep the better one ######################
                if latest_model_score <= older_model_score:
                    lg.exception(
                        f"The `latestly trained model` (with {latest_model_score} score)) ain't better than the older \
one (with {older_model_score} score) and quite evidently, the older model shall not be replaced!")
                    raise Exception(
                        f"The `latestly trained model` (with {latest_model_score} score)) ain't better than the older \
one (with {older_model_score} score) and quite evidently, the older model shall not be replaced!")

                lg.info("The `latestly trained model` (with {latest_model_score} score)) performed better than the \
older one and quite evidently, the older one shall be replaced!")

                ############################## Save Artifacts Config ##############################################
                model_eval_artifact = artifact.ModelEvaluationArtifact(
                    is_model_replaced=True, improved_accuracy=abs(older_model_score-latest_model_score)
                )
                
                lg.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            
            lg.info(f"Model Evaluation completed!")

            return model_eval_artifact
            ...
        except Exception as e:
            lg.exception(e)

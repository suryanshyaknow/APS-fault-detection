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
                ############## Load Older Model and respective Artifacts from Model Registry #################
                # fetch Model path
                lg.info(
                    "fetching the location of older Model from the Model Registry..")
                model_path = self.model_registry_config.get_latest_model_path()
                # fetch Transformer path
                lg.info(
                    "fetching the location of older corrosponding Transformer from the Model Registry..")
                transformer_path = self.model_registry_config.get_latest_transformer_path()
                # fetch Target Encoder path
                lg.info(
                    "fetching the location of older corrosponding Target Encoder from the Model Registry..")
                target_encoder_path = self.model_registry_config.get_latest_target_encoder_path()
                lg.info("locations fetched successfully!")
                # Loading them all
                lg.info("Now, loading the older model and corrosponding artifacts..")
                older_model = BasicUtils.load_object(
                    model_path, obj_desc="older Model")
                older_transformer = BasicUtils.load_object(
                    transformer_path, obj_desc="older Transformer")
                older_target_encoder = BasicUtils.load_object(
                    target_encoder_path, obj_desc="older Target Encoder")
                lg.info("Older Model and respective Artifacts fetched successfully!")

                #################### Load latest Model and corrosponding Artifacts ###########################
                lg.info("Now, loading the latest model and respective Arftifacts..")
                latest_model = BasicUtils.load_object(
                    self.model_training_artifact.model_path, obj_desc="latest Model")
                latest_transformer = BasicUtils.load_object(
                    self.data_transformation_artifact.transformer_path, obj_desc="latest Transformer")
                latest_target_encoder = BasicUtils.load_object(
                    self.data_transformation_artifact.target_encoder_path, obj_desc="latest Target Encoder")

                ########################## Load the test dataframe ###########################################
                lg.info("loading the test dataset from the `data ingestion artifact`..")
                # fetch the test dataframe
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
                # now fetch features and label separately
                X_test, y_test = BasicUtils.get_features_and_labels(df=test_df, target=[self.target], desc="Test")

                ######################## Evaluating the older Model's performance ############################
                # Transform the test features via the older Transformer
                X_test_arr = older_transformer.transform(X_test)
                # Encode the Categories to numerical dtype via the older Target Encoder
                y_true =older_target_encoder.transform(y_test)
                lg.info("Evaluating the performance of the `older model`..")
                lg.info(
                    "Making predictions on the test dataset using the `older model`..")
                y_pred = older_model.predict(X_test_arr)
                lg.info("predictions made successfully!")
                lg.info(
                    "Evaluating the performance of the `older model` using \"f1-score\"..")
                older_model_score = f1_score(y_true, y_pred)
                lg.info(f"Older Model's performance: {older_model_score}")

                ########################## Evaluating the latest Model's performance ###############################
                # Transform the test features via the latest Transformer
                X_test_arr = latest_transformer.transform(X_test)
                # Encode the Categories to numerical dtype via the latest Target Encoder
                y_true =latest_target_encoder.transform(y_test)
                lg.info("Evaluating the performance of the `latest model`..")
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

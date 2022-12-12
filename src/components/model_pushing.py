from src.entities.config import ModelPushingConfig
import os
from src.utils.file_operations import BasicUtils
from src.logger import lg
from src.entities.artifact import ModelPushingArtifact, ModelTrainingArtifact, DataTransformationArtifact
from src.CONFIG import ModelRegistryConfig
from dataclasses import dataclass


@dataclass
class ModelPushing:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelPushing" class')

    data_transformation_artifact: DataTransformationArtifact
    model_training_artifact: ModelTrainingArtifact

    model_pushing_config = ModelPushingConfig()
    model_registry_config = ModelRegistryConfig()

    def initiate(self) -> ModelPushingArtifact:
        try:
            lg.info(f"\n{'='*27} MODEL PUSHING {'='*40}")

            ############################# Load Objects which are to be saved ###################################
            # Load latest transformer from the `DataTransformation` artifact
            transformer_pipeline = BasicUtils.load_object(
                file_path=self.data_transformation_artifact.transformer_path, obj_desc="Transformer Pipeline")
            # Load latest model from the `ModelTraining` artifact
            model = BasicUtils.load_object(
                file_path=self.model_training_artifact.model_path, obj_desc="trained Model")

            ############################# Save them as `Model Pushing`` Artifacts ##############################
            lg.info(
                "Saving the \"model\" and the \"transformer pipline\" as the `Model Pushing` stage's artifacts..")
            # Save the Transformer Pipeline
            BasicUtils.save_object(
                file_path=self.model_pushing_config.to_be_pushed_transformer_path,
                obj=model,
                obj_desc="Transformer Pipeline"
            )
            # Save the Model
            BasicUtils.save_object(
                file_path=self.model_pushing_config.to_be_pushed_model_path,
                obj=transformer_pipeline,
                obj_desc="trained Model"
            )
            lg.info('`Model Pushing` stage\'s artifacts saved succesfully!')

            ############################# Save them to `Model Registry` dir ####################################
            lg.info(
                "Now, saving the \"model\" and the \"transformer pipline\" in the `Model Registry`..")
            # Save the Transformer Pipeline
            latest_transformer_dir = self.model_registry_config.save_latest_transformer_at()
            BasicUtils.save_object(
                file_path=latest_transformer_dir, obj=transformer_pipeline, obj_desc="Transformer Pipeline"
            )
            # Save the Model
            latest_model_dir = self.model_registry_config.save_latest_model_at()
            BasicUtils.save_object(
                file_path=latest_model_dir, obj=model, obj_desc="trained Model"
            )

            ############################# Save Artifacts Config ################################################
            model_pushing_artifact = ModelPushingArtifact(
                pushed_model_dir=self.model_pushing_config.to_be_pushed_model_path,
                saved_model_dir=latest_model_dir
            )

            lg.info(f"Model Pushing Artifact: {model_pushing_artifact}")
            lg.info(f"Model Pushing completed!")

            return model_pushing_artifact
            ...
        except Exception as e:
            lg.exception(e)

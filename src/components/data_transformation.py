import pandas as pd
import numpy as np
import os
from src.logger import lg
from src.utils.file_operations import BasicUtils
from src.entities.config import DataTransformationConfig, BaseConfig
from src.entities.artifact import DataIngestionArtifact, DataTransformationArtifact
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from typing import List
from imblearn.combine import SMOTETomek


@dataclass
class DataTransformation:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataTransformation" class')

    data_ingestion_artifact: DataIngestionArtifact
    data_transformation_config = DataTransformationConfig()
    target = BaseConfig().target

    @classmethod
    def get_transformer_pipeline(cls, num_atts: List, cat_atts: List) -> Pipeline:
        """Returns a `Custom Pipeline` for both numerical and categorical attributes of the said dataset.
        Pipeline contains `SimpleImputer` and `RobustScaler` for the numerical attributes and as such `OneHotEncoder`
        for the categorical ones.


        Args:
            num_atts (List): List of numerical attributes names.
            cat_atts (List): List of categorical attributes names. 

        Returns:
            Pipeline: Custom simultaneous Pipeline for the both numerical and categorical attributes. 
        """
        try:
            ########################## Pipeline for Numerical Atts ############################################
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()
            num_pipeline = Pipeline(
                steps=[("Imputer", simple_imputer),
                       ("Robust Scaler", robust_scaler)]
            )

            ########################## LabelEncoder for Categorical Atts ######################################
            enc = OneHotEncoder(drop="first", sparse="False", dtype="int64")

            ########################## Simultaneous Pipeline for both Atts ####################################
            transformer_pipeline = ColumnTransformer([
                ("Numerical Pipeline", num_pipeline, num_atts),
                ("OneHot Encoder", enc, cat_atts)
            ])

            return transformer_pipeline
            ...
        except Exception as e:
            lg.info(e)

    def initiate(self):
        try:
            lg.info(f"{'='*22} DATA INGESTION {'='*35}")

            ################## Fetch the Training and Test datasets ############################################
            lg.info("fetching the training and test sets for transformation..")
            training_set = pd.read_csv(
                self.data_ingestion_artifact.training_file_path)
            test_set = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            lg.info("training and test sets fetched successfully!")

            ################## Fetch the Numerical and Categorical attributes ##################################
            num_atts, cat_atts = BasicUtils.get_numerical_and_categorical_attributes(
                training_set, "Training")

            ################## Transformation using Transformer Pipeline #######################################
            # fetch the transformation pipeline and fit to the training set
            lg.info("fetching the transformation pipeline..")
            lg.info("fitting the transformation pipline to the training set..")
            transformation_pipline = DataTransformation.get_transformer_pipeline(
                num_atts=num_atts, cat_atts=cat_atts)
            transformation_pipline.fit(training_set)
            lg.info("Transformation Pipeline fitted successfully!")
            
            # Transformation of Training set
            lg.info('Transforming Training set..')
            training_set_transformed = pd.DataFrame(
                transformation_pipline.transform(training_set), columns=num_atts+cat_atts)
            lg.info("Training set transformed successfully..")
            # Transformation of Test set
            lg.info('Transforming Test set..')
            test_set_transformed = pd.DataFrame(
                transformation_pipline.transform(test_set), columns=num_atts+cat_atts)
            lg.info("Test set transformed successfully..")

            ################## Resampling of Data Instances ####################################################
            # Separate out features and target for resampling
            training_feats, training_target = BasicUtils.get_features_and_labels(
                training_set_transformed, target=[self.target], desc="Training"
            )
            test_feats, test_target = BasicUtils.get_features_and_labels(
                test_set_transformed, target=[self.target], desc="Test"
            )

            lg.info(
                "Resampling the data instances as our target attribute is highly imbalanced..")
            lg.info(
                f"Before Resampling, shape of the `training set`: {training_set_transformed.shape}")
            lg.info(
                f"Before Resampling, shape of the `test set`: {test_set_transformed.shape}")
            lg.info('Resampling via SMOTETomek using sampling_strategy="minority"..')
            smt_tomek = SMOTETomek(sampling_strategy="minority")
            training_arr_feats_res, training_arr_target_res = smt_tomek.fit_resample(
                training_feats, training_target)
            test_arr_feats_res, test_arr_target_res = smt_tomek.fit_resample(
                test_feats, test_target)

            lg.info("resampling of both training and test sets done successfully!")

            ################## Configure Training and Test arrays' features and labels #########################
            training_arr_res = np.c_[
                training_arr_feats_res, training_arr_target_res]
            lg.info(
                f"After Resampling, shape of the `training set`: {training_arr_res.shape}")
            test_arr_res = np.c_[test_arr_feats_res, test_arr_target_res]
            lg.info(
                f"After Resampling, shape of the `test set`: {test_arr_res.shape}")

            ################### Save Training and Test arrays ##################################################
            # Saving the Training Array
            BasicUtils.save_numpy_array(
                file_path=self.data_transformation_config.transformed_training_file_path,
                arr=training_arr_res,
                desc="Training"
            )
            # Saving the Test Array
            BasicUtils.save_numpy_array(
                file_path=self.data_transformation_config.transformed_test_file_path,
                arr=test_arr_res,
                desc="Test"
            )

            ########################## Save Artifacts Config ###################################################
            transformation_artifact = DataTransformationArtifact(
                transformer_path=self.data_transformation_config.transformer_path,
                transformed_training_file_path=self.data_transformation_config.transformed_training_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return transformation_artifact
            ...
        except Exception as e:
            lg.exception(e)

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
from dataclasses import dataclass
from imblearn.combine import SMOTETomek


@dataclass
class DataTransformation:
    """Shall be used for preprocessing and transformation of data before making any statistical analyses and feeding 
    the data into ML algorithms.

    Args:
        data_ingestion_artifact (DataIngestionArtifact): Takes in a `DataIngestionArtifact` object as pre-requisite to 
        trigger the Data Transformation stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataTransformation" class')

    data_ingestion_artifact: DataIngestionArtifact
    data_transformation_config = DataTransformationConfig()
    target = BaseConfig().target

    @classmethod
    def get_transformer(cls) -> Pipeline:
        """Returns a `Custom Pipeline` for numerical attributes of the said dataset. Pipeline contains 
        `SimpleImputer` and `RobustScaler` to transform the features of the very same dataset.

        Raises:
            e: Raises relevant exception should any sort of error pops up while fetching the said `transformer pipeline`.

        Returns:
            Pipeline: Custom Pipeline for the numerical features of the said dataset. 
        """
        try:
            ################################# Pipeline for Numerical Atts #####################################
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            robust_scaler = RobustScaler()
            transformer = Pipeline(
                steps=[("Imputer", simple_imputer),
                       ("Robust Scaler", robust_scaler)])

            return transformer
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    @classmethod
    def get_target_encoder(cls) -> OneHotEncoder:
        """Returns the OneHotEncoder to transform the categories of the target column into the numerical dtype.

        Raises:
            e: Raises relevant exception should any sort of error pops up while fetching the said `target encoder`.

        Returns:
            OneHotEncoder: OneHotEncoder to encode and decode the categories of the target column when desired.
        """
        try:
            onehot_enc = OneHotEncoder(
                drop="first", sparse=False, dtype="int64")
            return onehot_enc
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def initiate(self) -> DataTransformationArtifact:
        """Initiates the Data Transformation stage of the training pipeline and returns the configurations of relevant artifacts
        (being used in the process) and transformed datasets (being generated in the process).

        Raises:
            e: Raises relevant exception should any sort of error pops up in the Data Transformation stage.

        Returns:
            DataTransformationArtifact: Contains configurations of `transformer pipeline`, `target encoder` and transformed 
            training and test arrays. 
        """
        try:
            lg.info(f"\n{'='*27} DATA TRANSFORMATION {'='*40}")

            ############################# Fetch the Training and Test datasets ################################
            lg.info("fetching the training and test sets for transformation..")
            training_set = pd.read_csv(
                self.data_ingestion_artifact.training_file_path)
            test_set = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            lg.info("training and test sets fetched successfully!")

            ################ Fetch the Features and Labels from the Training and Test sets #####################
            X_train, y_train = BasicUtils.get_features_and_labels(
                df=training_set, target=[self.target], desc="Training set")
            X_test, y_test = BasicUtils.get_features_and_labels(
                df=test_set, target=[self.target], desc="Test set")

            ######################### Transformation using Transformer and Encoder ############################
            # fetch the transformer and fit to the training set
            lg.info("fetching the transformer..")
            lg.info("fitting the transformer to the Training set's features..")
            transformer = DataTransformation.get_transformer()
            transformer.fit(X_train)
            lg.info("Transformer fitted to the \"Training features\" successfully!")
            # fetch the Encoder and fit to the Target column
            lg.info("fetching the target encoder..")
            lg.info("fitting the target encoder to the Target column..")
            target_enc = DataTransformation.get_target_encoder()
            target_enc.fit(y_train)
            lg.info("Target Encoder fitted to the \"Target column\" successfully!")

            # Transformation of Training set's features and target
            lg.info('Transforming Training features..')
            X_train_transformed = transformer.transform(X_train)
            lg.info("Encoding Target column of the training set..")
            y_train_encoded = target_enc.transform(y_train)

            # Transformation of Test set's features and target
            lg.info('Transforming Test features..')
            X_test_transformed = transformer.transform(X_test)
            lg.info("Encoding Target column of the test set..")
            y_test_encoded = target_enc.transform(y_test)

            ########################## Save the Transformer and the TargetEncoder #############################
            # Saving the Transformer
            BasicUtils.save_object(
                file_path=self.data_transformation_config.transformer_path,
                obj=transformer,
                obj_desc="transformer")
            # Saving the Target Encoder
            BasicUtils.save_object(
                file_path=self.data_transformation_config.target_encoder_path,
                obj=target_enc,
                obj_desc="target encoder")

            ############################### Resampling of Training Instances ##################################
            lg.info(
                "Resampling the training instances as our target attribute is highly imbalanced..")
            lg.info(
                f"Before Resampling, shape of the `training set`: {training_set.shape}")
            lg.info('Resampling via SMOTETomek using sampling_strategy="auto"..')
            smt_tomek = SMOTETomek(sampling_strategy="auto")
            X_train_res, y_train_res = smt_tomek.fit_resample(
                X_train_transformed, y_train_encoded)

            lg.info("..resampling of training instances is done successfully!")

            ############################# Configure Training and Test arrays ##################################
            training_arr_res = np.c_[X_train_res, y_train_res]
            lg.info(
                f"After Resampling, shape of the `training set`: {training_arr_res.shape}")
            test_arr_res = np.c_[X_test_transformed, y_test_encoded]
            lg.info(
                f"Configured the training and test arrays successfully!")

            ################################ Save Training and Test arrays ####################################
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

            ##################################### Save Artifacts Config #######################################
            transformation_artifact = DataTransformationArtifact(
                transformer_path=self.data_transformation_config.transformer_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path,
                transformed_training_file_path=self.data_transformation_config.transformed_training_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            lg.info(f"Transformation Artifact: {transformation_artifact}")
            lg.info("Data Transformation completed!")

            return transformation_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
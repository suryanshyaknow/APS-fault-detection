import os
import pandas as pd
import numpy as np
import argparse
from src.logger import lg
from src.CONFIG import ModelRegistryConfig
from src.utils.file_operations import BasicUtils
from datetime import datetime
from dataclasses import dataclass

PREDICTION_DIR = "predictions"


@dataclass
class BatchPredictionPipeline:
    """Shall be used for triggering the prediction pipeline.

    Args:
        input file path (str): Location of batch input file for which predictions gotta be made.
    """
    lg.info("Prediction Pipeline commences now..")
    lg.info(
        f"Entered the {os.path.basename(__file__)[:-3]}.BatchPredictionPipeline")

    input_file_path: str
    model_registry_config = ModelRegistryConfig()

    def get_predicition_file_path(self) -> str:
        """Returns the file path where the Predictions file is to be stored. And generates a new one in regard 
        to the datetime stamp, each time this function is called.
        
        Raises:
            e: Throws exception should any error or exception pops up while execution of this method.

        Returns:
            str: Path where prepared prediction file's gotta be stored.
        """
        try:
            # Create Prediction dir if not already there
            os.makedirs(PREDICTION_DIR, exist_ok=True)
            # Add datetimestamp followed by double underscore in front of the input file name to generate the name of Prediction file
            prediction_file = os.path.basename(self.input_file_path).replace(
                ".csv", f"__{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
            prediction_file_path = os.path.join(
                PREDICTION_DIR, prediction_file)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            return prediction_file_path

    def initiate(self) -> str:
        """Triggers the prediction pipeline flow, making predictions for the input batch file and returns the 
        prepared prediction file.

        Raises:
            e: Throws exception should any error or exception pops up while execution of the prediction pipeline.

        Returns:
            str: Location of the prepared prediction file.
        """
        try:
            ############## Read the dataset from the given path on which prediction is to be done ##############
            lg.info(
                f"fetching the data from the input file at \"{self.input_file_path}\"")
            input_df = pd.read_csv(self.input_file_path)
            lg.info("data fetched as Dataframe successfully!")
            lg.info(f"Shape of the data fetched: {input_df.shape}")
            # Replace all `na` values with np.NaN
            input_df.replace({"na": np.NaN}, inplace=True)

            ######################## Load the Transformer and Transform the input data #########################
            # Load the Transformer from the Model Registry
            lg.info("loading the \"Transformer\" from the Model Registry..")
            transformer = BasicUtils.load_object(
                file_path=self.model_registry_config.get_latest_transformer_path(), obj_desc="Transformer")
            lg.info("transforming the input data..")
            # first and foremost fetch the features that were used in training
            input_features = list(transformer.feature_names_in_)
            # transform the input features from the input file and compose the consequent array
            input_arr = transformer.transform(input_df[input_features])
            lg.info(f"Input data transformed successfully!")

            ############################## Load the Model and Make Predictions #################################
            # Load the Model
            lg.info(
                "loading the latestly \"trained model\" from the Model Registry, for making predictions..")
            model = BasicUtils.load_object(
                file_path=self.model_registry_config.get_latest_model_path(), obj_desc="latestly trained Model")
            lg.info("Making predictions..")
            preds = model.predict(input_arr).reshape(-1, 1)
            lg.info("Predictions made successfully!")

            ##################### Load the Encoder and Inverse-transform the Predictions #######################
            # Grab the `OneHot Encoder` to inverse-transform predictions
            lg.info(
                "fetching the `OneHot Encoder` from the Model registry to inverse transform the predcitions..")
            target_enc = BasicUtils.load_object(
                file_path=self.model_registry_config.get_latest_target_encoder_path(), obj_desc="Target Encoder")
            lg.info(
                f"fitted OneHot Encoder: {target_enc} fetched successfully!")
            # Inverse transform the predictions
            lg.info("Inverse-transforming the predictions..")
            cat_preds = target_enc.inverse_transform(preds)
            lg.info("predictions transformed successfully!")

            ##################### Configure the Predicitons and Save the Predictions file ######################
            # Configure the Categorical Predictions into the dataframe
            input_df["prediction"] = cat_preds
            # Save the Prediction file
            prediction_file_path = self.get_predicition_file_path()
            lg.info("Readying the prediction file..")
            input_df.to_csv(
                path_or_buf=self.get_predicition_file_path(), index=None)
            ...
        except Exception as e:
            lg.exception(e)
            raise e
        else:
            lg.info(f'Prediction file\'s ready at "{prediction_file_path}"')
            return prediction_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path", default="aps_failure_training_set1.csv")
    parsed_args = parser.parse_args()
    prediction_pipeline = BatchPredictionPipeline(
        input_file_path=parsed_args.input_file_path)
    prediction_pipeline.initiate()

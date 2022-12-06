import pandas as pd
import numpy as np
import os
from src.logger import lg
from scipy.stats import ks_2samp
from typing import Optional
from dataclasses import dataclass
from src.entities.config import DataValidationConfig
from src.entities.artifact import DataValidationArtifact, DataIngestionArtifact
from src.utils.file_operations import BasicUtils


@dataclass
class DataValidation:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.DataValidation" class')

    data_validation_config: DataValidationConfig
    data_ingestion_artifact: DataIngestionArtifact
    basic_utils = BasicUtils()
    validation_report = dict()

    def drop_redundant_columns(self, df: pd.DataFrame, missing_thresh: float, report_key: str) -> Optional[pd.DataFrame]:
        """Drops the columns having missing values more than said threshold.

        Args:
            df (pd.DataFrame): Accepts the dataframe whose columns have to be dropped.
            missing_thresh (float): Percentage criterion to drop a column.

        Returns:
            Optional[pd.DataFrame]: Dataframe after getting its redundant columns droppeds.
        """
        try:
            lg.info(
                f"Dropping columns having missing values more than {missing_thresh*100}%..")
            cols_missing_ratios = df.isna().sum().div(df.shape[0])
            cols_to_drop = list(
                cols_missing_ratios[cols_missing_ratios > missing_thresh].index)

            if len(cols_to_drop) == 0:
                lg.info("None of the columns needs to be dropped!")
            else:
                lg.info(f"Columns to be dropped: {cols_to_drop}")
                self.validation_report[report_key] = cols_to_drop

            # Dropping redundant columns
            df.drop(cols_to_drop, axis=1, inplace=True)

            if len(df.columns) == 0:
                lg.info(
                    "Now that all columns ended up getting dropped, dataframe contains NADA!")
                return None
            return df
            ...
        except Exception as e:
            lg.exception(e)

    def required_columns_check(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key: str) -> bool:
        """Performs check for columns taking reference as the MDM (Master Data Management).

        Args:
            base_df (pd.DataFrame): Reference dataframe.
            current_df (pd.DataFrame): Present dataframe on which check has to be performed having `base_df` as reference.
            report_key (str): Key name for holding dropped columns in the validation report.
        Returns:
            bool: True if all the required columns are present inside our current dataframe, else False.
        """
        try:
            lg.info(
                "Validating columns from the reference MDM (Master Data Management)..")

            base_cols = base_df.columns
            current_cols = current_df.columns

            missing_cols = []
            for col in base_cols:
                if col not in current_cols:
                    lg.info(f'"{col}" column is missing')
                    missing_cols.append(col)

            if len(missing_cols) > 0:
                self.validation_report[report_key] = missing_cols
                return False
            return True
            ...
        except Exception as e:
            lg.exception(e)        

    def data_drift_check(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key: str) -> None:
        """Performs "data drift" check by validating if the distributions of both base dataframe and present 
        dataframe are drawn from a single distribution by assuming Null Hypothesis as in they are indeed drawn 
        from the same distribution.

        Args:
            base_df (pd.DataFrame): Reference Dataframe
            current_df (pd.DataFrame): Dataframe on which data drift check has to be done.
            report_key (str): Key name for holding missing columns in the validation report.
        """
        try:
            ##################### Configuring Columns' datatypes ##############################################
            base_df = self.basic_utils.configure_float_columns(base_df, exclude_columns=["class"])
            current_df = self.basic_utils.configure_float_columns(current_df, exclude_columns=["class"])

            drift_report = {}

            ##################### Separating Columns ##########################################################    
            base_num_cols, base_cat_cols = self.basic_utils.separate_numerical_and_categorical_columns(base_df)
            
            ##################### DRIFT CHECK for Numerical Columns ###########################################
            lg.info(
                'Performing "Data Drift Check" for "Numerical Columns" by validating if the distributions of both base dataframe and present dataframe are drawn from a single distribution..'
            )
            lg.info('Null Hypothesis: Base datframe\'s distribution and current dataframe\'s distribution are drawn from a single distribution.')
            for col in base_num_cols:
                base_dist, current_dist = base_df[col], current_df[col]
                kstest_result = ks_2samp(base_dist, current_dist)
                
                lg.info(f'Null Hypothesis: "{col}" from base_df and "{col}" from current_df are drawn from the same distribution.')
                if kstest_result.pvalue > .05:
                    lg.info("Null hypothesis is to be accepted!")
                    drift_report[col] = {
                        "pvalue": float(kstest_result.pvalue),
                        "same_distribution": True
                    }
                else:
                    
                    lg.info("Null hypothesis is to be rejected!")
                    drift_report[col] = {
                        "pvalue": float(kstest_result.pvalue),
                        "same_distribution": False
                    }

            ##################### DRIFT CHECK for Categorical Columns #########################################
            lg.info(
                'Performing "Data Drift Check" for "Categorical Columns" by validating whether the categories present inside both of the dataframe\'s categorical columns are same..')
            for col in base_cat_cols:
                if list(base_df[col].value_counts().index).sort() == list(current_df[col].value_counts().index).sort():
                    drift_report[col] = {
                        "same_categories": True
                    }
                else:
                    drift_report[col] = {
                        "same_categories": False
                    }
            
            ##################### Drift Report -> Validation Report ############################################
            self.validation_report[report_key] = drift_report
            ...
        except Exception as e:
            lg.exception(e)

    def initiate(self) -> DataValidationArtifact:
        try:
            lg.info(f"{'='*22} DATA VALIDATION {'='*35}")

            lg.info("fetching Base dataframe..")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            # Replace na vals with np.NaN
            base_df.replace({"na": np.NaN}, inplace=True)
            ############################ DROPPING COLUMNS #####################################################
            base_df = self.drop_redundant_columns(
                base_df, missing_thresh=self.data_validation_config.missing_thresh,
                report_key="dropped_columns_from_base_data")

            lg.info("fetching Training dataframe..")
            train_df = pd.read_csv(self.data_ingestion_artifact.training_file_path)
            lg.info("dropping columns from the training data..")
            train_df = self.drop_redundant_columns(
                train_df, missing_thresh=self.data_validation_config.missing_thresh,
                report_key="dropped_columns_from_training_data"
            )
            lg. info("fetching Test dataframe..")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            lg.info("dropping columns from the test data..")
            test_df = self.drop_redundant_columns(
                test_df, missing_thresh=self.data_validation_config.missing_thresh,
                report_key="dropped_columns_from_test_data"
            )

            ############################ REQUIRED COLUMN CHECK ###############################################
            lg.info('"Required Columns Check" for the training dataset..')
            train_columns_status = self.required_columns_check(
                base_df, train_df, "missing_columns_in_training_data")
            lg.info('"Required Columns Check" for the test dataset..')
            test_columns_status = self.required_columns_check(
                base_df, test_df, "missing_columns_in_test_data")
            # If the "Required Column Check" for the given dataset passes then only, 
            # "Data Drift Check" can be performed.
            if train_columns_status:
                lg.info("Since all required columns are there in the training set, now going for the \"Data Drift Check\"..")
                self.data_drift_check(base_df, train_df, "data_drift_within_training_data")
            if test_columns_status:
                lg.info("Since all required columns are there in the test set, now going for the \"Data Drift Check\"..")
                self.data_drift_check(base_df, test_df, "data_drift_within_test_data")

            ###################### Dumping VALIDATION REPORT into a YAML file ###############################
            lg.info("Dumping Validation Report inside yaml file..")
            self.basic_utils.write_yaml_file(
                file_path=self.data_validation_config.report_file_path, 
                data=self.validation_report,
                desc="Validation Report")

            ###################### Saving ARTIFACTS Config ##################################################
            data_validation_artifact = DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path
            )
            lg.info(f"Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
            ...
        except Exception as e:
            lg.exception(e)
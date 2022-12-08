import os
from src.logger import lg
from datetime import datetime
from dataclasses import dataclass


RAW_DATA_FILE = "aps_failure_training_set1.csv"
FEATURE_STORE_FILE = "sensors.csv"
TRAINING_FILE = "training_set.csv"
TEST_FILE = "test_set.csv"
TRANSFORMER_PIPELINE = "transformer_pipeline.pkl"


@dataclass
class BaseConfig:
    project: str = "APS-fault-detection"
    target: str = "class"


@dataclass
class DataSourceConfig:
    database_name: str = "sensors-streaming-data"
    collection_name: str = "APS_sensors"
    raw_data_path: str = os.path.join(os.getcwd(), RAW_DATA_FILE)


@dataclass
class TrainingPipelineConfig:
    try:
        artifact_dir: str = os.path.join(
            os.getcwd(), "artifacts", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
    except Exception as e:
        lg.exception(e)


class DataIngestionConfig:
    def __init__(self) -> None:
        try:
            training_pipeline_config = TrainingPipelineConfig()
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir, "data_ingestion")

            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir, FEATURE_STORE_FILE)
            self.training_file_path = os.path.join(
                self.data_ingestion_dir, TRAINING_FILE)
            self.test_file_path = os.path.join(
                self.data_ingestion_dir, TEST_FILE)
            self.test_size = 0.2
            self.random_state = 42
        except Exception as e:
            lg.exception(e)


class DataValidationConfig:
    def __init__(self) -> None:
        try:
            training_pipeline_config = TrainingPipelineConfig()
            self.data_validation_dir = os.path.join(
                training_pipeline_config.artifact_dir, "data_validation")

            self.base_file_path = os.path.join(os.getcwd(), RAW_DATA_FILE)
            self.missing_thresh = .3
            self.report_file_path = os.path.join(
                self.data_validation_dir, "report.yaml")
            ...
        except Exception as e:
            lg.exception(e)


class DataTransformationConfig:
    def __init__(self) -> None:
        try:
            training_pipeline_config = TrainingPipelineConfig()
            self.data_transformation_dir = os.path.join(
                training_pipeline_config.artifact_dir, "data_transformation")

            self.transformer_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_PIPELINE)
            self.transformed_training_file_path = os.path.join(
                self.data_transformation_dir, TRAINING_FILE.replace(".csv", ".npz"))
            self.transformed_test_file_path = os.path.join(
                self.data_transformation_dir, TEST_FILE.replace(".csv", ".npz"))
            ...
        except Exception as e:
            lg.exception(e)

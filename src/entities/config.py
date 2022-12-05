import os
from src.logger import lg
from datetime import datetime
from dataclasses import dataclass


RAW_DATA_FILE = "aps_failure_training_set1.csv"
FEATURE_STORE_FILE = "sensors_feature_store.csv"
TRAINING_FILE = "training_set.csv"
TEST_FILE = "test_set.csv"


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
    def __init__(self, training_pipeline_config:TrainingPipelineConfig=TrainingPipelineConfig()) -> None:
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "data ingestion") 
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, FEATURE_STORE_FILE) 
            self.training_file_path = os.path.join(self.data_ingestion_dir, TRAINING_FILE)
            self.test_file_path = os.path.join(self.data_ingestion_dir, TEST_FILE)
            self.test_size = 0.2
            self.random_state = 42
        except Exception as e:
            lg.exception(e)


class Datavalidation:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig) -> None:
        try:
            self.data_validation_dir = os.path.joijn(training_pipeline_config.artifact_dir, "data validation")
            self.base_file_path = os.path.join(os.getcwd(), RAW_DATA_FILE)
            self.missing_thresh = .3
            self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
            ...
        except Exception as e:
            lg.exception(e)    

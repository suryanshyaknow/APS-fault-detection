from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    feature_store_file: str
    training_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformer_path: str
    transformed_training_file_path: str
    transformed_test_file_path: str


@dataclass
class ModelTrainerArtifact:
    ...


@dataclass
class ModelEvaluationArtifact:
    ...


@dataclass
class ModelPusherArtifact:
    ...

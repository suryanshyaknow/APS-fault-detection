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
class DataTransformation:
    ...


@dataclass
class ModelTrainerArtifact:
    ...


@dataclass
class ModelEvaluationArtifact:
    ...


@dataclass
class ModelPusherArtifact:
    ...

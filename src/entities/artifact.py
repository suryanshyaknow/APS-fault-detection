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
    target_encoder_path: str
    transformed_training_file_path: str
    transformed_test_file_path: str


@dataclass
class ModelTrainingArtifact:
    model_path: str
    f1_training_score: float
    f1_test_score: float


@dataclass
class ModelEvaluationArtifact:
    is_model_replaced: bool
    improved_accuracy: float


@dataclass
class ModelPushingArtifact:
    pushed_model_dir: str
    saved_model_dir: str

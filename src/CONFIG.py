import os
from dotenv import load_dotenv
from src.logger import lg
from typing import Optional
from dataclasses import dataclass
from src.entities.config import MODEL_FILE, TRANSFORMER, TARGET_ENCODER


@dataclass
class Config:
    """Helps to access the secret and protected params declared inside the .env that cannot be exposed beyond 
    the scope of the project.
    """
    load_dotenv()
    mongodb_url: str = os.getenv("MONGO_DB_URL")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECERET_ACCESS_KEY")


class ModelRegistryConfig:
    def __init__(
            self, model_registry: str = "saved_models", transformer_dir: str = "transformer",
            target_encoder_dir: str = "encoder", model_dir: str = "model") -> None:

        self.model_registry = model_registry
        # Making sure the Model Registry does exist
        os.makedirs(self.model_registry, exist_ok=True)

        self.transformer_dir = transformer_dir
        self.target_encoder_dir = target_encoder_dir
        self.model_dir = model_dir
        ...

    def get_latest_dir_path(self) -> Optional[str]:
        """Returns path of the latest dir of Model Registry.

        Returns:
            Optional[str]: Path of the latest dir of Model Registry.
        """
        try:
            dirs = os.listdir(self.model_registry)
            if len(dirs) == 0:
                lg.warning(
                    "As of now there are no such directories in the Model Registry!")
                return None

            # Typecasting dir names from str to int
            dirs = list(map(int, dirs))
            latest_dir = max(dirs)

            return os.path.join(self.model_registry, str(latest_dir))
            ...
        except Exception as e:
            lg.exception(e)

    def get_latest_model_path(self) -> str:
        """Returns the path of the latest `model` dir of the Model Registry.

        Returns:
            str: Latest Model dir path of the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info("Getting the `latest model path` from the Model Registry..")
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a model, shame!")

            return os.path.join(latest_dir, self.model_dir, MODEL_FILE)
            ...
        except Exception as e:
            lg.exception(e)

    def get_latest_transformer_path(self) -> str:
        """Returns the path of the latest `transformer` dir of the Model Registry.

        Returns:
            str: Latest Transformer dir path of the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info("Getting the `latest transformer path` from the Model Registry..")
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a transformer, shame!")

            return os.path.join(latest_dir, self.transformer_dir, TRANSFORMER)
            ...
        except Exception as e:
            lg.exception(e)
    
    def get_latest_target_encoder_path(self) -> str:
        """Returns the path of the latest `target encoder` dir of the Model Registry.

        Returns:
            str: Latest Target Encoder dir path of the Model Registry.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info("Getting the `latest Target Encoder path` from the Model Registry..")
            if latest_dir is None:
                lg.exception(
                    "Even the dir doesn't exist and you are expecting a target encoder, shame!")

            return os.path.join(latest_dir, self.target_encoder_dir, TARGET_ENCODER)
            ...
        except Exception as e:
            lg.exception(e)

    def get_latest_dir_path_to_save(self) -> str:
        """Returns the latest dir where the latest models and relevant artifacts can be stored.

        Returns:
            str: Latest dir path to save the latest models and relevant artifacts.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info("Configuring the dir path where the `latest artifacts` are to be saved..")

            if latest_dir is None:
                return os.path.join(self.model_registry, str(0))
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, str(latest_dir_num+1))
            ...
        except Exception as e:
            lg.exception(e)

    def save_latest_transformer_at(self) -> str:
        """Dir path in the Model Registry to save the latest Transformer at.

        Returns:
            str: Dir path where the latest Transformer is to be stored.
        """
        try:
            latest_dir_to_save = self.get_latest_dir_path_to_save()
            lg.info("Configuring the path where the `latestly built Transformer` is to be stored..")
            return os.path.join(latest_dir_to_save, self.transformer_dir, TRANSFORMER)
            ...
        except Exception as e:
            lg.exception(e)

    def save_latest_target_encoder_at(self) -> str:
        """Dir path in the Model Registry to save the latest Target Encoder at.

        Returns:
            str: Dir path where the latest Target Encoder is to be stored.
        """
        try:
            latest_dir_to_save = self.get_latest_dir_path()
            lg.info("Configuring the path where the `latestly built Target Encoder` is to be stored..")
            return os.path.join(latest_dir_to_save, self.target_encoder_dir, TARGET_ENCODER)
            ...
        except Exception as e:
            lg.exception(e)

    def save_latest_model_at(self) -> str:
        """Dir path in the Model Registry to save the latest Model at.

        Returns:
            str: Dir path where the latest Model is to be stored.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            lg.info("Configuring the path where the `latestly trained Model` is to be stored..")
            return os.path.join(latest_dir, self.model_dir, MODEL_FILE)
            ...
        except Exception as e:
            lg.exception(e)


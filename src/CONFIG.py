import os
from dotenv import load_dotenv
from src.logger import lg
from dataclasses import dataclass


@dataclass
class Config:
    """Helps to access the secret and protected params declared inside the .env that cannot be exposed beyond 
    the scope of the project.
    """
    load_dotenv()
    mongodb_url: str = os.getenv("MONGO_DB_URL")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = os.getenv("AWS_SECERET_ACCESS_KEY")
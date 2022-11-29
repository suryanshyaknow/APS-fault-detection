import logging as lg
import os
from datetime import datetime


# Name of the file where the logs are to be recorded
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"

# Logs directory
LOG_FILE_DIR = os.path.join(os.getcwd(), ".logs")

# Create the dir if not there already
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Log file path
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# Logs' configuration
lg.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=lg.INFO
)
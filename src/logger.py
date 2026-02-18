import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs",log_file)    # Path to the FOLDER
os.makedirs(logs_dir, exist_ok=True)

log_file_path = os.path.join(logs_dir, log_file)  # Path to the FILE

logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,  
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

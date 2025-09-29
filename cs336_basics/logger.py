import logging
import os
from datetime import datetime
import rich

from cs336_basics.common_data import ExperimentConfig

is_setup = False
logger = logging.getLogger("cs336_basics")

def print(message: str) -> None:
    rich.print(message)
    if is_setup: 
        logger.info(message)


def setup_logging(exp_config: ExperimentConfig) -> None:
    global is_setup
    if is_setup:
        return
    is_setup = True
    log_path = exp_config.log_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if exp_config.name == "":
        exp_config.name = "default"
    log_file_name = f"log_{exp_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = os.path.join(log_path, log_file_name)
    print(f"Training log will be saved to {log_file}")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
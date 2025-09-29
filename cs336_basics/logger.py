import logging
import os
from datetime import datetime
import rich
from cs336_basics.common_data import ExperimentConfig

def print_and_log(message: str) -> None:
    rich.print(message)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.info(message)


def setup_logging(exp_config: ExperimentConfig, log_file_name: str | None = None) -> str:
    log_path = exp_config.log_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if log_file_name is None:
        log_file_name = f"log_{exp_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = os.path.join(log_path, log_file_name)
    print_and_log(f"Training log will be saved to {log_file}")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return log_file_name
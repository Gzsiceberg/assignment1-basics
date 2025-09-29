import logging
import os
from datetime import datetime
import rich

is_setup = False
logger = logging.getLogger("cs336_basics")

def print(message: str) -> None:
    rich.print(message)
    if is_setup: 
        logger.info(message)


def setup_logging(args) -> None:
    global is_setup
    if is_setup:
        return
    is_setup = True
    log_path = "logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_name = os.path.basename(args.file)
    file_name_no_ext = os.path.splitext(file_name)[0]
    log_file_name = f"log_{file_name_no_ext}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = os.path.join(log_path, log_file_name)
    print(f"Training log will be saved to {log_file}")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # log all arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("-" * 120)

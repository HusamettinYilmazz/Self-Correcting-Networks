import os
import sys
import logging
import torch.distributed as dist
from datetime import datetime


class Logger:
    def __init__(self, save_dir):
        self.is_main = (not dist.is_initialized()) or dist.get_rank() == 0

        self.save_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%B_%d___%H_%M_%S")
        self.log_file = os.path.join(self.save_dir, f"train_{timestamp}.log")

        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"kaggle_logger")
        logger.setLevel(logging.INFO)

        # IMPORTANT: avoid duplicate handlers in DDP
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s]: %(message)s',
            datefmt='%Y %B %d %H:%M:%S'
        )

        # file logging only on rank 0
        if self.is_main:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def info(self, message):
        if self.is_main:
            self.logger.info(message)

    def warning(self, message):
        if self.is_main:
            self.logger.warning(message)

    def error(self, message):
        if self.is_main:
            self.logger.error(message)
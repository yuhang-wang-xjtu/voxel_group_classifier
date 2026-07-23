import logging


def get_root_logger(logger_name="pointcept", log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger

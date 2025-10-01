import logging
from logging import Logger


def configure_logging(logger: Logger, log_output: str):
    ignored_dependencies = ['transitions']

    for dependency in ignored_dependencies:
        logging.getLogger(dependency).setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_output),
            logging.StreamHandler()
        ]
    )

    return logger

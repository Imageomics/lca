import logging  # NOQA
import logging.config

LOGGING_FORMAT = '%(levelname)-6s %(asctime)s [%(filename)18s:%(lineno)3d] %(message)s'
LOGGING_LEVEL = logging.INFO

def get_formatter():
    return logging.Formatter(LOGGING_FORMAT)


def reset_logging():
    logger = logging.getLogger("beta_stability")
    print(f"Resetting loggers: {logger}")
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    logger.handlers.clear()

def init_logger():
    print("INIT")
    reset_logging()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    formatter = get_formatter()
    handler = logging.StreamHandler()
    handler.setLevel(LOGGING_LEVEL)
    handler.setFormatter(formatter)

    logger = logging.getLogger("beta_stability")
    logger.setLevel(LOGGING_LEVEL)
    logger.propagate = 0
    logger.addHandler(handler)
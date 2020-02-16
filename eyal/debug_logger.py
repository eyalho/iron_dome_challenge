import logging
import os
import time


class OnlyDebugFilter(logging.Filter):
    """remove warnings, log only debug.."""

    def filter(self, record):
        return record.levelno == logging.DEBUG


def create_logger(name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler, file handler and set level to debug
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    directory = "logs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{logger.name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}.log")

    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # add ch to logger
    f = OnlyDebugFilter()
    logger.addFilter(f)

    # def debug(msg, *args, **kwargs):
    #     return logging.debug(msg, *args, **kwargs)
    return logger


# test logger
if __name__ == "__main__":
    logger = create_logger("test")
    logger.debug('debug message')
    logger.debug('parsing debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical messag')

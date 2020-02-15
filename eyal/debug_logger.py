import logging


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

    fh = logging.FileHandler("debug.log")
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

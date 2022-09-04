from contextlib import contextmanager
from logging import getLogger, DEBUG, INFO, FileHandler, Formatter, Logger, StreamHandler
import time
from typing import Optional


def get_logger(filepath: str, name: Optional[str] = None) -> Logger:
    """Get logger having stream and file handler.

    Parameters
    ----------
    filepath : str
        Where log file is to be written.
    name : Optional[str], optional
        Logger name, by default None.

    Returns
    -------
    Logger: Logger
        Logger object.
    """
    logger = getLogger(name or __name__)
    logger.setLevel(DEBUG)
    for h in logger.handlers:
        logger.removeHandler(h)
    file_handler = FileHandler(filepath)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter('"%(asctime)s","%(name)s","%(levelname)s","%(message)s"'))
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@contextmanager
def timer(name: str, logger: Optional[Logger] = None, level: int = DEBUG):
    '''
    Refference
    ----------
    https://amalog.hateblo.jp/entry/kaggle-snippets
    '''
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'{name}: start')
    yield
    print_(f'{name}: done in {time.time() - t0:.3f} s')
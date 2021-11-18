import logging
import time

from functools import wraps
import inspect

import numpy as np


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def create_custom_logger(level: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("logger instantiated")
    return logger


def log_function_call(logger: logging.Logger):

    def log_this(function):

        @wraps(function)
        def logging_wrapper_function(*args, **kwargs):

            logger.info(f'Started: {function.__name__}')
            args_repr = [repr(a) if not isinstance(a, np.ndarray) else repr(a.shape) for a in args]
            kwargs_repr = [f"{k}={v}" if not isinstance(v, np.ndarray) else f"{k} shape={v.shape}"
                           for k, v in kwargs.items()]
            default_params = [f"{k}={v}" for k, v in get_default_args(function).items()]
            signature_ = "\n".join(args_repr + kwargs_repr + default_params)
            logger.info(f"function {function.__name__} called with args {signature_}")
            output = function(*args, **kwargs)
            if not isinstance(output, np.ndarray):
                logger.info(f"output of {function.__name__}: {output}")
            else:
                logger.info(f"output shape of {function.__name__}: {output.shape}")
            return output

        return logging_wrapper_function

    return log_this


def timer(func):

    @wraps(func)
    def wrapper_function(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end-start} sec(s) to run.')
        return result

    return wrapper_function



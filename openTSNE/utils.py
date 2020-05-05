from functools import wraps
from time import time
import warnings


class Timer:
    def __init__(self, message, verbose=False):
        self.message = message
        self.start_time = time()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print("===>", self.message)

    def __exit__(self, *args):
        end_time = time()
        if self.verbose:
            print("   --> Time elapsed: %.2f seconds" % (end_time - self.start_time))


def deprecate_parameter(parameter):
    def wrapper(f):
        @wraps(f)
        def func(*args, **kwargs):
            if parameter in kwargs:
                warnings.warn(
                    f"The parameter `{parameter}` has been deprecated and will be "
                    f"removed in future versions",
                    category=FutureWarning,
                )
            return f(*args, **kwargs)
        return func
    return wrapper

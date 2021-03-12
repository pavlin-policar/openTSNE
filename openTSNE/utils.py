from functools import wraps
from time import time
import warnings
import numpy as np


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


def is_package_installed(libname):
    """Check whether a python package is installed."""
    import importlib

    try:
        importlib.import_module(libname)
        return True
    except ImportError:
        return False


def clip_point_to_disc(points, radius, inplace=False):
    if not inplace:
        points = points.copy()

    r = np.linalg.norm(points, axis=1)
    phi = np.arctan2(points[:, 0], points[:, 1])
    mask = r > radius
    np.clip(r, 0, radius, out=r)
    points[:, 0] = r * np.sin(phi)
    points[:, 1] = r * np.cos(phi)

    return points, mask

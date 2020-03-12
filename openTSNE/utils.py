from time import time


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

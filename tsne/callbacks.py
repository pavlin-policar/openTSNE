import time


class ErrorLogger:
    def __init__(self):
        self.iter_count = 0
        self.last_log_time = None

    def __call__(self, iteration, error, embedding):
        # Initialize values in first iteration
        if iteration == 1:
            self.iter_count = 0
            self.last_log_time = time.time()
            return True

        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        n_iters = iteration - self.iter_count
        self.iter_count = iteration

        print('Iteration % 4d, KL divergence % 6.4f, %d iterations in %.4f sec' % (
            iteration, error, n_iters, duration))

        return True

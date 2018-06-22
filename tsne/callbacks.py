import time


class ErrorLogger:
    def __init__(self):
        self.iter_count = 0
        self.last_log_time = time.time()

    def __call__(self, iteration, error, embedding):
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        # Reset iter count if we run optimization multiple times
        if iteration < self.iter_count:
            self.iter_count = 0

        n_iters = iteration - self.iter_count
        self.iter_count = iteration

        print('Iteration % 4d, KL divergence % 6.4f, %d iterations in %.4f sec' % (
            iteration, error, n_iters, duration))

        return True

import logging
import time
from functools import partial

import numpy as np

from tsne import kl_divergence
from tsne.tsne import TSNEEmbedding

log = logging.getLogger(__name__)


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


class VerifyExaggerationError:
    """Used to verify that the exaggeration correction implemented in
    `gradient_descent` is correct."""
    def __init__(self, embedding: TSNEEmbedding) -> None:
        self.embedding = embedding
        # Keep a copy of the unexaggerated affinity matrix
        self.P = self.embedding.P.copy()

    def __call__(self, iteration: int, corrected_error: float, embedding: TSNEEmbedding):
        params = self.embedding.gradient_descent_params
        method = params['negative_gradient_method']

        if np.sum(embedding.P) <= 1:
            log.warning('Are you sure you are testing an exaggerated P matrix?')

        if method == 'fft':
            f = partial(kl_divergence.kl_divergence_approx_fft,
                        n_interpolation_points=params['n_interpolation_points'],
                        min_num_intervals=params['min_num_intervals'],
                        ints_in_interval=params['ints_in_interval'])
        elif method == 'bh':
            f = partial(kl_divergence.kl_divergence_approx_bh, theta=params['theta'])

        P = self.P

        true_error = f(P.indices, P.indptr, P.data, embedding)
        if abs(true_error - corrected_error) > 1e-8:
            raise RuntimeError('Correction term is wrong.')
        else:
            log.info('Corrected: %.4f - True %.4f [eps %.4f]' % (
                corrected_error, true_error, abs(true_error - corrected_error)))

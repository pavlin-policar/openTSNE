import logging
import time
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix

from . import kl_divergence
from .tsne import TSNEEmbedding

log = logging.getLogger(__name__)


class Callback:
    def optimization_about_to_start(self):
        """This is called at the beginning of the optimization procedure."""

    def __call__(self, iteration, error, embedding):
        """This is the main method called from the optimization.

        Parameters
        ----------
        iteration: int
            The current iteration number.

        error: float
            The current KL divergence of the given embedding.

        embedding: TSNEEmbedding
            The current t-SNE embedding.

        Returns
        -------
        stop_optimization: bool
            If this value is set to ``True``, the optimization will be
            interrupted.

        """


class ErrorLogger(Callback):
    """Basic error logger.

    This logger prints out basic information about the optimization. These
    include the iteration number, error and how much time has elapsed from the
    previous callback invocation.

    """

    def __init__(self):
        self.iter_count = 0
        self.last_log_time = None

    def optimization_about_to_start(self):
        self.last_log_time = time.time()
        self.iter_count = 0

    def __call__(self, iteration, error, embedding):
        now = time.time()
        duration = now - self.last_log_time
        self.last_log_time = now

        n_iters = iteration - self.iter_count
        self.iter_count = iteration

        print("Iteration % 4d, KL divergence % 6.4f, %d iterations in %.4f sec" % (
            iteration, error, n_iters, duration))


class VerifyExaggerationError(Callback):
    """Used to verify that the exaggeration correction implemented in
    `gradient_descent` is correct."""
    def __init__(self, embedding: TSNEEmbedding) -> None:
        self.embedding = embedding
        # Keep a copy of the unexaggerated affinity matrix
        self.P = self.embedding.affinities.P.copy()

    def __call__(self, iteration: int, corrected_error: float, embedding: TSNEEmbedding):
        params = self.embedding.gradient_descent_params
        method = params["negative_gradient_method"]

        if np.sum(embedding.affinities.P) <= 1:
            log.warning("Are you sure you are testing an exaggerated P matrix?")

        if method == "fft":
            f = partial(kl_divergence.kl_divergence_approx_fft,
                        n_interpolation_points=params["n_interpolation_points"],
                        min_num_intervals=params["min_num_intervals"],
                        ints_in_interval=params["ints_in_interval"])
        elif method == "bh":
            f = partial(kl_divergence.kl_divergence_approx_bh, theta=params["theta"])

        P = self.P

        true_error = f(P.indices, P.indptr, P.data, embedding)
        if abs(true_error - corrected_error) > 1e-8:
            raise RuntimeError("Correction term is wrong.")
        else:
            log.info("Corrected: %.4f - True %.4f [eps %.4f]" % (
                corrected_error, true_error, abs(true_error - corrected_error)))


class ErrorApproximations(Callback):
    """Check how good the error approximations are. Of course, we use an
    approximation for P so this itself is an approximation."""
    def __init__(self, P: csr_matrix):
        self.P = P.copy()
        self.exact_errors = []
        self.bh_errors = []
        self.fft_errors = []

    def __call__(self, iteration: int, error: float, embedding: TSNEEmbedding):
        exact_error = kl_divergence.kl_divergence_exact(self.P.toarray(), embedding)
        bh_error = kl_divergence.kl_divergence_approx_bh(
            self.P.indices, self.P.indptr, self.P.data, embedding)
        fft_error = kl_divergence.kl_divergence_approx_fft(
            self.P.indices, self.P.indptr, self.P.data, embedding)

        self.exact_errors.append(exact_error)
        self.bh_errors.append(bh_error)
        self.fft_errors.append(fft_error)

    def report(self):
        exact_errors = np.array(self.exact_errors)
        bh_errors = np.array(self.bh_errors)
        fft_errors = np.array(self.fft_errors)

        bh_diff = bh_errors - exact_errors
        print("Barnes-Hut: mean difference %.4f (±%.4f)" % (
            np.mean(bh_diff), np.std(bh_diff)))

        fft_diff = fft_errors - exact_errors
        print("Interpolation: mean difference %.4f (±%.4f)" % (
            np.mean(fft_diff), np.std(fft_diff)))

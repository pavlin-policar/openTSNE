import inspect
import logging
import multiprocessing
from collections import Iterable
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator

from . import _tsne
from . import initialization as initialization_scheme
from .affinity import Affinities, PerplexityBasedNN
from .quad_tree import QuadTree

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


def _check_callbacks(callbacks):
    if callbacks is not None:
        # If list was passed, make sure all of them are actually callable
        if isinstance(callbacks, Iterable):
            if any(not callable(c) for c in callbacks):
                raise ValueError("`callbacks` must contain callable objects!")
        # The gradient descent method deals with lists
        elif callable(callbacks):
            callbacks = (callbacks,)
        else:
            raise ValueError("`callbacks` must be a callable object!")

    return callbacks


def _handle_nice_params(optim_params: dict) -> None:
    """Convert the user friendly params into something the optimizer can
    understand."""
    # Handle callbacks
    optim_params["callbacks"] = _check_callbacks(optim_params.get("callbacks"))
    optim_params["use_callbacks"] = optim_params["callbacks"] is not None

    # Handle negative gradient method
    negative_gradient_method = optim_params.pop("negative_gradient_method")
    if callable(negative_gradient_method):
        negative_gradient_method = negative_gradient_method
    elif negative_gradient_method in {"bh", "BH", "barnes-hut"}:
        negative_gradient_method = kl_divergence_bh
    elif negative_gradient_method in {"fft", "FFT", "interpolation"}:
        negative_gradient_method = kl_divergence_fft
    else:
        raise ValueError("Unrecognized gradient method. Please choose one of "
                         "the supported methods or provide a valid callback.")
    # `gradient_descent` uses the more informative name `objective_function`
    optim_params["objective_function"] = negative_gradient_method

    # Handle number of jobs
    n_jobs = optim_params.get("n_jobs", 1)
    if n_jobs < 0:
        n_cores = multiprocessing.cpu_count()
        # Add negative number of n_jobs to the number of cores, but increment by
        # one because -1 indicates using all cores, -2 all except one, and so on
        n_jobs = n_cores + n_jobs + 1

    # If the number of jobs, after this correction is still <= 0, then the user
    # probably thought they had more cores, so we'll default to 1
    if n_jobs <= 0:
        log.warning("`n_jobs` receieved value %d but only %d cores are available. "
                    "Defaulting to single job." % (optim_params["n_jobs"], n_cores))
        n_jobs = 1

    optim_params["n_jobs"] = n_jobs


def __check_init_num_samples(num_samples, required_num_samples):
    if num_samples != required_num_samples:
        raise ValueError(
            "The provided initialization contains a different number "
            "of points (%d) than the data provided (%d)." % (
                num_samples, required_num_samples)
        )


def __check_init_num_dimensions(num_dimensions, required_num_dimensions):
    if num_dimensions != required_num_dimensions:
        raise ValueError(
            "The provided initialization contains a different number "
            "of components (%d) than the embedding (%d)." % (
                num_dimensions, required_num_dimensions)
        )


init_checks = SimpleNamespace(
    num_samples=__check_init_num_samples,
    num_dimensions=__check_init_num_dimensions,
)


class OptimizationInterrupt(InterruptedError):
    """Optimization was interrupted by a callback.

    Parameters
    ----------
    error: float
        The KL divergence of the embedding.

    final_embedding: Union[TSNEEmbedding, PartialTSNEEmbedding]
        Is either a partial or full embedding, depending on where the error was
        raised.

    """
    
    def __init__(self, error, final_embedding):
        super().__init__()
        self.error = error
        self.final_embedding = final_embedding


class PartialTSNEEmbedding(np.ndarray):
    """A partial t-SNE embedding.

    A partial embedding is created when we take an existing
    :class:`TSNEEmbedding` and embed new samples into the embedding space. It
    differs from the typical embedding in that it is not possible to add new
    samples to a partial embedding and would generally be a bad idea.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    embedding: np.ndarray
        Initial positions for each data point.

    reference_embedding: TSNEEmbedding
        The embedding into which the new samples are to be added.

    P : array_like
        An :math:`N \\times M` affinity matrix containing the affinities from each new
        data point :math:`n` to each data point in the existing embedding
        :math:`m`.

    learning_rate: float
        The learning rate for t-SNE optimization. Typical values range between
        100 to 1000. Setting the learning rate too low or too high may result in
        the points forming a "ball". This is also known as the crowding problem.

    exaggeration: float
        The exaggeration factor is used to increase the attractive forces of
        nearby points, producing more compact clusters.

    momentum: float
        Momentum accounts for gradient directions from previous iterations,
        resulting in faster convergence.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``.

    theta: float
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    callbacks: Callable[[int, float, np.ndarray] -> bool]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    optimizer: gradient_descent
        Optionally, an existing optimizer can be used for optimization. This is
        useful for keeping momentum gains between different calls to
        :func:`optimize`.

    Attributes
    ----------
    kl_divergence: float
        The KL divergence or error of the embedding.

    """

    def __new__(cls, embedding, reference_embedding, P, optimizer=None,
                **gradient_descent_params):
        init_checks.num_samples(embedding.shape[0], P.shape[0])

        obj = np.asarray(embedding, dtype=np.float64, order="C").view(PartialTSNEEmbedding)

        obj.reference_embedding = reference_embedding
        obj.P = P
        obj.gradient_descent_params = gradient_descent_params

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, gradient_descent):
            raise TypeError("`optimizer` must be an instance of `%s`, but got `%s`." % (
                gradient_descent.__class__.__name__, type(optimizer)))
        obj.optimizer = optimizer

        obj.kl_divergence = None

        return obj

    def optimize(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):
        """Run optmization on the embedding for a given number of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: float
            The learning rate for t-SNE optimization. Typical values range
            between 100 to 1000. Setting the learning rate too low or too high
            may result in the points forming a "ball". This is also known as the
            crowding problem.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        random_state: Union[int, RandomState]
            The random state parameter follows the convention used in
            scikit-learn. If the value is an int, random_state is the seed used
            by the random number generator. If the value is a RandomState
            instance, then it will be used as the random number generator. If
            the value is None, the random number generator is the RandomState
            instance used by `np.random`.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        PartialTSNEEmbedding
            An optimized partial t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = PartialTSNEEmbedding(
                np.copy(self),
                self.reference_embedding,
                self.P,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        _handle_nice_params(optim_params)
        optim_params["n_iter"] = n_iter

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding,
                reference_embedding=self.reference_embedding,
                P=self.P,
                **optim_params,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


class TSNEEmbedding(np.ndarray):
    """A t-SNE embedding.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    embedding: np.ndarray
        Initial positions for each data point.

    affinities: Affinities
        An affinity index which can be used to compute the affinities of new
        points to the points in the existing embedding. The affinity index also
        contains the affinity matrix :math:`P` used during optimization.

    learning_rate: float
        The learning rate for t-SNE optimization. Typical values range between
        100 to 1000. Setting the learning rate too low or too high may result in
        the points forming a "ball". This is also known as the crowding problem.

    exaggeration: float
        The exaggeration factor is used to increase the attractive forces of
        nearby points, producing more compact clusters.

    momentum: float
        Momentum accounts for gradient directions from previous iterations,
        resulting in faster convergence.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``.

    theta: float
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    random_state: Union[int, RandomState]
        The random state parameter follows the convention used in scikit-learn.
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    callbacks: Callable[[int, float, np.ndarray] -> bool]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    optimizer: gradient_descent
        Optionally, an existing optimizer can be used for optimization. This is
        useful for keeping momentum gains between different calls to
        :func:`optimize`.

    Attributes
    ----------
    kl_divergence: float
        The KL divergence or error of the embedding.

    """

    def __new__(cls, embedding, affinities, random_state=None, optimizer=None,
                **gradient_descent_params):
        init_checks.num_samples(embedding.shape[0], affinities.P.shape[0])

        obj = np.asarray(embedding, dtype=np.float64, order="C").view(TSNEEmbedding)

        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.random_state = random_state

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, gradient_descent):
            raise TypeError("`optimizer` must be an instance of `%s`, but got `%s`." % (
                gradient_descent.__class__.__name__, type(optimizer)))
        obj.optimizer = optimizer

        obj.kl_divergence = None

        return obj

    def optimize(self, n_iter, inplace=False, propagate_exception=False,
                 **gradient_descent_params):
        """Run optmization on the embedding for a given number of steps.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: float
            The learning rate for t-SNE optimization. Typical values range
            between 100 to 1000. Setting the learning rate too low or too high
            may result in the points forming a "ball". This is also known as the
            crowding problem.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        random_state: Union[int, RandomState]
            The random state parameter follows the convention used in
            scikit-learn. If the value is an int, random_state is the seed used
            by the random number generator. If the value is a RandomState
            instance, then it will be used as the random number generator. If
            the value is None, the random number generator is the RandomState
            instance used by `np.random`.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        TSNEEmbedding
            An optimized t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = TSNEEmbedding(
                np.copy(self),
                self.affinities,
                random_state=self.random_state,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        _handle_nice_params(optim_params)
        optim_params["n_iter"] = n_iter

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding, P=self.affinities.P, **optim_params
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding

    def transform(self, X, perplexity=5, initialization="median", k=25,
                  learning_rate=1, n_iter=100, exaggeration=2, momentum=0):
        """Embed new points into the existing embedding.

        This procedure optimizes each point only with respect to the existing
        embedding i.e. it ignores any interactions between the points in ``X``
        among themselves.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be added to the existing embedding.

        perplexity: float
            Perplexity can be thought of as the continuous :math:`k` number of
            nearest neighbors, for which t-SNE will attempt to preserve
            distances. However, when transforming, we only consider neighbors in
            the existing embedding i.e. each data point is placed into the
            embedding, independently of other new data points.

        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``median``, ``weighted`` or
            ``random``. In all cases, ``median`` of ``weighted`` should be
            preferred.

        k: int
            The number of nearest neighbors to consider when initially placing
            the point onto the embedding. This is different from ``perpelxity``
            because perplexity affects optimization while this only affects the
            initial point positions.

        learning_rate: float
            The learning rate for t-SNE optimization. Typical values range
            between 100 to 1000. Setting the learning rate too low or too high
            may result in the points forming a "ball". This is also known as the
            crowding problem.

        n_iter: int
            The number of iterations to run in the normal optimization regime.
            Typically, the number of iterations needed when adding new data
            points is much lower than with regular optimization.

        exaggeration: float
            The exaggeration factor to use during the normal optimization phase.
            This can be used to form more densely packed clusters and is useful
            for large data sets.

        momentum: float
            The momentum to use during optimization phase.

        Returns
        -------
        PartialTSNEEmbedding
            The positions of the new points in the embedding space.

        """

        # We check if the affinity `to_new` methods takes the `perplexity`
        # parameter and raise an informative error if not. This happes when the
        # user uses a non-standard affinity class e.g. multiscale, then attempts
        # to add points via `transform`. These classes take `perplexities` and
        # fail
        affinity_signature = inspect.signature(self.affinities.to_new)
        if "perplexity" not in affinity_signature.parameters:
            raise TypeError(
                "`transform` currently does not support non `%s` type affinity "
                "classes. Please use `prepare_partial` and `optimize` to add "
                "points to the embedding." % PerplexityBasedNN.__name__
            )

        embedding = self.prepare_partial(
            X, perplexity=perplexity, initialization=initialization, k=k
        )

        try:
            embedding.optimize(
                n_iter=n_iter,
                learning_rate=learning_rate,
                exaggeration=exaggeration,
                momentum=momentum,
                inplace=True,
                propagate_exception=True,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            embedding = ex.final_embedding

        return embedding

    def prepare_partial(self, X, initialization="median", k=25, **affinity_params):
        """Prepare a partial embedding which can be optimized.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be added to the existing embedding.

        initialization: Union[np.ndarray, str]
            The initial point positions to be used in the embedding space. Can
            be a precomputed numpy array, ``median``, ``weighted`` or
            ``random``. In all cases, ``median`` of ``weighted`` should be
            preferred.

        k: int
            The number of nearest neighbors to consider when initially placing
            the point onto the embedding. This is different from ``perpelxity``
            because perplexity affects optimization while this only affects the
            initial point positions.

        **affinity_params: dict
            Additional params to be passed to the ``Affinities.to_new`` method.
            Please see individual :class:`~openTSNE.affinity.Affinities`
            implementations as the parameters differ between implementations.

        Returns
        -------
        PartialTSNEEmbedding
            An unoptimized :class:`PartialTSNEEmbedding` object, prepared for
            optimization.

        """
        P, neighbors, distances = self.affinities.to_new(
            X, return_distances=True, **affinity_params
        )

        # If initial positions are given in an array, use a copy of that
        if isinstance(initialization, np.ndarray):
            init_checks.num_samples(initialization.shape[0], X.shape[0])
            init_checks.num_dimensions(initialization.shape[1], self.shape[1])

            embedding = np.array(initialization)

        # Random initialization with isotropic normal distribution
        elif initialization == "random":
            embedding = initialization_scheme.random(X, self.shape[1], self.random_state)
        elif initialization == "weighted":
            embedding = initialization_scheme.weighted_mean(
                X, self, neighbors[:, :k], distances[:, :k]
            )
        elif initialization == "median":
            embedding = initialization_scheme.median(self, neighbors[:, :k])
        else:
            raise ValueError(f"Unrecognized initialization scheme `{initialization}`.")

        return PartialTSNEEmbedding(
            embedding,
            reference_embedding=self,
            P=P,
            **self.gradient_descent_params,
        )


class TSNE(BaseEstimator):
    """t-Distributed Stochastic Neighbor Embedding.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    n_components: int
        The dimension of the embedding space. This deafults to 2 for easy
        visualization, but sometimes 1 is used for t-SNE heatmaps. t-SNE is
        not designed to embed into higher dimension and please note that
        acceleration schemes break down and are not fully implemented.

    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        nearest neighbors, for which t-SNE will attempt to preserve distances.

    learning_rate: float
        The learning rate for t-SNE optimization. Typical values range between
        100 to 1000. Setting the learning rate too low or too high may
        result in the points forming a "ball". This is also known as the
        crowding problem.

    early_exaggeration_iter: int
        The number of iterations to run in the *early exaggeration* phase.

    early_exaggeration: float
        The exaggeration factor to use during the *early exaggeration* phase.
        Typical values range from 12 to 32.

    n_iter: int
        The number of iterations to run in the normal optimization regime.

    exaggeration: float
        The exaggeration factor to use during the normal optimization phase.
        This can be used to form more densely packed clusters and is useful
        for large data sets.

    theta: float
        Only used when ``negative_gradient_method="bh"`` or its other aliases.
        This is the trade-off parameter between speed and accuracy of the tree
        approximation method. Typical values range from 0.2 to 0.8. The value 0
        indicates that no approximation is to be made and produces exact results
        also producing longer runtime.

    n_interpolation_points: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The number of interpolation points to use within each grid cell for
        interpolation based t-SNE. It is highly recommended leaving this value
        at the default 3.

    min_num_intervals: int
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        The minimum number of grid cells to use, regardless of the
        ``ints_in_interval`` parameter. Higher values provide more accurate
        gradient estimations.

    ints_in_interval: float
        Only used when ``negative_gradient_method="fft"`` or its other aliases.
        Indicates how large a grid cell should be e.g. a value of 3 indicates a
        grid side length of 3. Lower values provide more accurate gradient
        estimations.

    initialization: Union[np.ndarray, str]
        The initial point positions to be used in the embedding space. Can be a
        precomputed numpy array, ``pca`` or ``random``. Please note that when
        passing in a precomputed positions, it is highly recommended that the
        point positions have small variance (var(Y) < 0.0001), otherwise you may
        get poor embeddings.

    metric: str
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    initial_momentum: float
        The momentum to use during the *early exaggeration* phase.

    final_momentum: float
        The momentum to use during the normal optimization phase.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    neighbors: str
        Specifies the nearest neighbor method to use. Can be either ``exact`` or
        ``approx``.

    negative_gradient_method: str
        Specifies the negative gradient approximation method to use. For smaller
        data sets, the Barnes-Hut approximation is appropriate and can be set
        using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
        For larger data sets, the FFT accelerated interpolation method is more
        appropriate and can be set using one of the following aliases: ``fft``,
        ``FFT`` or ``ìnterpolation``.

    callbacks: Union[Callable, List[Callable]]
        Callbacks, which will be run every ``callbacks_every_iters`` iterations.

    callbacks_every_iters: int
        How many iterations should pass between each time the callbacks are
        invoked.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    """

    def __init__(self, n_components=2, perplexity=30, learning_rate=200,
                 early_exaggeration_iter=250, early_exaggeration=12,
                 n_iter=750, exaggeration=None,
                 theta=0.5, n_interpolation_points=3, min_num_intervals=10,
                 ints_in_interval=1, initialization="pca", metric="euclidean",
                 metric_params=None, initial_momentum=0.5, final_momentum=0.8,
                 n_jobs=1, neighbors="approx", negative_gradient_method="fft",
                 callbacks=None, callbacks_every_iters=50, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.n_iter = n_iter
        self.exaggeration = exaggeration
        self.theta = theta
        self.n_interpolation_points = n_interpolation_points
        self.min_num_intervals = min_num_intervals
        self.ints_in_interval = ints_in_interval

        # Check if the number of components match the initialization dimension
        if isinstance(initialization, np.ndarray):
            init_checks.num_dimensions(initialization.shape[1], n_components)
        self.initialization = initialization

        self.metric = metric
        self.metric_params = metric_params
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_jobs = n_jobs
        self.neighbors_method = neighbors
        self.negative_gradient_method = negative_gradient_method

        self.callbacks = callbacks
        self.callbacks_every_iters = callbacks_every_iters

        self.random_state = random_state

    def fit(self, X):
        """Fit a t-SNE embedding for a given data set.

        Runs the standard t-SNE optimization, consisting of the early
        exaggeration phase and a normal optimization phase.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be embedded.

        Returns
        -------
        TSNEEmbedding
            A fully optimized t-SNE embedding.

        """
        embedding = self.prepare_initial(X)

        try:
            # Early exaggeration with lower momentum to allow points to find more
            # easily move around and find their neighbors
            embedding.optimize(
                n_iter=self.early_exaggeration_iter,
                exaggeration=self.early_exaggeration,
                momentum=self.initial_momentum,
                inplace=True,
                propagate_exception=True,
            )

            # Restore actual affinity probabilities and increase momentum to get
            # final, optimized embedding
            embedding.optimize(
                n_iter=self.n_iter,
                exaggeration=self.exaggeration,
                momentum=self.final_momentum,
                inplace=True,
                propagate_exception=True,
            )

        except OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            embedding = ex.final_embedding

        return embedding

    def prepare_initial(self, X):
        """Prepare the initial embedding which can be optimized as needed.

        Parameters
        ----------
        X: np.ndarray
            The data matrix to be embedded.

        Returns
        -------
        TSNEEmbedding
            An unoptimized :class:`TSNEEmbedding` object, prepared for
            optimization.

        """
        # If initial positions are given in an array, use a copy of that
        if isinstance(self.initialization, np.ndarray):
            init_checks.num_samples(self.initialization.shape[0], X.shape[0])
            init_checks.num_dimensions(self.initialization.shape[1], self.n_components)

            embedding = np.array(self.initialization)

            variance = np.var(embedding, axis=0)
            if any(variance > 1e-4):
                log.warning(
                    "Variance of embedding is greater than 0.0001. Initial "
                    "embeddings with high variance may have display poor convergence."
                )

        elif self.initialization == "pca":
            embedding = initialization_scheme.pca(
                X, self.n_components, random_state=self.random_state
            )
        elif self.initialization == "random":
            embedding = initialization_scheme.random(
                X, self.n_components, random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unrecognized initialization scheme `{self.initialization}`."
            )

        affinities = PerplexityBasedNN(
            X,
            self.perplexity,
            method=self.neighbors_method,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        gradient_descent_params = {
            # Degrees of freedom of the Student's t-distribution. The
            # suggestion degrees_of_freedom = n_components - 1 comes from [3]_.
            "dof":  max(self.n_components - 1, 1),

            "negative_gradient_method": self.negative_gradient_method,
            "learning_rate": self.learning_rate,
            # By default, use the momentum used in unexaggerated phase
            "momentum": self.final_momentum,

            # Barnes-Hut params
            "theta": self.theta,
            # Interpolation params
            "n_interpolation_points": self.n_interpolation_points,
            "min_num_intervals": self.min_num_intervals,
            "ints_in_interval": self.ints_in_interval,

            "n_jobs": self.n_jobs,
            # Callback params
            "callbacks": self.callbacks,
            "callbacks_every_iters": self.callbacks_every_iters,
        }

        return TSNEEmbedding(
            embedding,
            affinities=affinities,
            random_state=self.random_state,
            **gradient_descent_params,
        )


def kl_divergence_bh(embedding, P, dof, bh_params, reference_embedding=None,
                     should_eval_error=False, n_jobs=1, **_):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = _tsne.estimate_negative_gradient_bh(
        tree, embedding, gradient, **bh_params, dof=dof, num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def kl_divergence_fft(embedding, P, dof, fft_params, reference_embedding=None,
                      should_eval_error=False, n_jobs=1, **_):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # Compute negative gradient.
    if embedding.ndim == 1 or embedding.shape[1] == 1:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d_with_reference(
                embedding.ravel(), reference_embedding.ravel(), gradient.ravel(),
                **fft_params,
            )
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_1d(
                embedding.ravel(), gradient.ravel(), **fft_params,
            )
    elif embedding.shape[1] == 2:
        if reference_embedding is not None:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d_with_reference(
                embedding, reference_embedding, gradient, **fft_params,
            )
        else:
            sum_Q = _tsne.estimate_negative_gradient_fft_2d(
                embedding, gradient, **fft_params,
            )
    else:
        raise RuntimeError(
            "Interpolation based t-SNE for >2 dimensions is currently "
            "unsupported (and generally a bad idea)"
        )

    # The positive gradient function needs a reference embedding always
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices, P.indptr, P.data, embedding, reference_embedding, gradient,
        dof, num_threads=n_jobs, should_eval_error=should_eval_error,
    )

    gradient *= 2 * (dof + 1) / dof
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


class gradient_descent:
    def __init__(self):
        self.gains = None

    def copy(self):
        optimizer = self.__class__()
        if self.gains is not None:
            optimizer.gains = np.copy(self.gains)
        return optimizer

    def __call__(self, embedding, P, n_iter, objective_function, learning_rate=200,
                 momentum=0.5, exaggeration=None, dof=1, min_gain=0.01,
                 min_grad_norm=1e-8, theta=0.5, n_interpolation_points=3,
                 min_num_intervals=10, ints_in_interval=1, reference_embedding=None,
                 n_jobs=1, use_callbacks=False, callbacks=None, callbacks_every_iters=50):
        """Perform batch gradient descent with momentum and gains.

        Parameters
        ----------
        embedding: np.ndarray
            The embedding :math:`Y`.

        P: array_like
            Joint probability matrix :math:`P`.

        n_iter: int
            The number of iterations to run for.

        objective_function: Callable[..., Tuple[float, np.ndarray]]
            A callable that evaluates the error and gradient for the current
            embedding.

        learning_rate: float
            The learning rate for t-SNE optimization. Typical values range
            between 100 to 1000. Setting the learning rate too low or too high
            may result in the points forming a "ball". This is also known as the
            crowding problem.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        dof: float
            Degrees of freedom of the Student's t-distribution.

        min_gain: float
            Minimum individual gain for each parameter.

        min_grad_norm: float
            If the gradient norm is below this threshold, the optimization will
            be stopped.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        ints_in_interval: float
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. Indicates how large a grid cell should be e.g. a value of 3
            indicates a grid side length of 3. Lower values provide more
            accurate gradient estimations.

        reference_embedding: np.ndarray
            If we are adding points to an existing embedding, we have to compute
            the gradients and errors w.r.t. the existing embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        use_callbacks: bool

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        float
            The KL divergence of the optimized embedding.
        np.ndarray
            The optimized embedding Y.

        Raises
        ------
        OptimizationInterrupt
            If the provided callback interrupts the optimization, this is raised.

        """
        assert isinstance(embedding, np.ndarray), \
            "`embedding` must be an instance of `np.ndarray`. Got `%s` instead" \
            % type(embedding)

        if reference_embedding is not None:
            assert isinstance(reference_embedding, np.ndarray), \
                "`reference_embedding` must be an instance of `np.ndarray`. Got " \
                "`%s` instead" % type(reference_embedding)

        update = np.zeros_like(embedding)
        if self.gains is None:
            self.gains = np.ones_like(embedding)

        bh_params = {"theta": theta}
        fft_params = {"n_interpolation_points": n_interpolation_points,
                      "min_num_intervals": min_num_intervals,
                      "ints_in_interval": ints_in_interval}

        # Lie about the P values for bigger attraction forces
        if exaggeration is None:
            exaggeration = 1

        if exaggeration != 1:
            P *= exaggeration

        # Notify the callbacks that the optimization is about to start
        if isinstance(callbacks, Iterable):
            for callback in callbacks:
                # Only call function if present on object
                getattr(callback, "optimization_about_to_start", lambda: ...)()

        for iteration in range(n_iter):
            should_call_callback = use_callbacks and (iteration + 1) % callbacks_every_iters == 0
            should_eval_error = should_call_callback

            error, gradient = objective_function(
                embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
                reference_embedding=reference_embedding, n_jobs=n_jobs,
                should_eval_error=should_eval_error,
            )

            # Correct the KL divergence w.r.t. the exaggeration if needed
            if should_eval_error and exaggeration != 1:
                error = error / exaggeration - np.log(exaggeration)

            if should_call_callback:
                # Continue only if all the callbacks say so
                should_stop = any((bool(c(iteration + 1, error, embedding)) for c in callbacks))
                if should_stop:
                    # Make sure to un-exaggerate P so it's not corrupted in future runs
                    if exaggeration != 1:
                        P /= exaggeration
                    raise OptimizationInterrupt(error=error, final_embedding=embedding)

            # Update the embedding using the gradient
            grad_direction_flipped = np.sign(update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            self.gains[grad_direction_flipped] += 0.2
            self.gains[grad_direction_same] = self.gains[grad_direction_same] * 0.8 + min_gain
            update = momentum * update - learning_rate * self.gains * gradient
            embedding += update

            # Zero-mean the embedding only if we're not adding new data points,
            # otherwise this will reset point positions
            if reference_embedding is None:
                embedding -= np.mean(embedding, axis=0)

            if np.linalg.norm(gradient) < min_grad_norm:
                log.info("Gradient norm eps reached. Finished.")
                break

        # Make sure to un-exaggerate P so it's not corrupted in future runs
        if exaggeration != 1:
            P /= exaggeration

        # The error from the loop is the one for the previous, non-updated
        # embedding. We need to return the error for the actual final embedding, so
        # compute that at the end before returning
        error, _ = objective_function(
            embedding, P, dof=dof, bh_params=bh_params, fft_params=fft_params,
            reference_embedding=reference_embedding, n_jobs=n_jobs,
            should_eval_error=True,
        )

        return error, embedding

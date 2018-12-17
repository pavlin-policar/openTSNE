import inspect
import logging
import unittest
from functools import wraps, partial
from typing import Callable, Any, Tuple, Optional
from unittest.mock import patch, MagicMock

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import openTSNE
from openTSNE import affinity, initialization
from openTSNE.affinity import PerplexityBasedNN
from openTSNE.nearest_neighbors import NNDescent
from openTSNE.tsne import kl_divergence_bh, kl_divergence_fft

np.random.seed(42)
affinity.log.setLevel(logging.ERROR)

TSNE = partial(openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh")


def check_params(params: dict) -> Callable:
    """Run a series of parameterized tests to check tSNE parameter flow."""

    def _decorator(test_case: Callable) -> Callable:
        @wraps(test_case)
        def _wrapper(self):
            for param_name in params:
                for param_value in params[param_name]:
                    test_case(self, param_name, param_value)

        return _wrapper

    return _decorator


def check_call_contains_kwargs(
        call: Tuple,
        params: dict,
        param_mapping: Optional[dict] = None,
) -> None:
    """Check whether a `call` object was called with some params, but also some
    others we don't care about"""
    _param_mapping = {"negative_gradient_method": "objective_function",
                      "early_exaggeration_iter": "n_iter",
                      "late_exaggeration_iter": "n_iter",
                      "early_exaggeration": "exaggeration",
                      "late_exaggeration": "exaggeration",
                      "initial_momentum": "momentum",
                      "final_momentum": "momentum"}
    if param_mapping is not None:
        _param_mapping.update(param_mapping)

    name, args, kwargs = call
    for key in params:
        # If a parameter isn't named the same way in the call
        if key in _param_mapping:
            kwargs_key = _param_mapping[key]
        else:
            kwargs_key = key

        expected_value = params[key]
        actual_value = kwargs.get(kwargs_key, None)
        if expected_value != actual_value:
            raise AssertionError(
                "Mock not called with `%s=%s`. Called with `%s`" %
                (key, expected_value, actual_value)
            )


def check_mock_called_with_kwargs(mock: MagicMock, params: dict) -> None:
    """Check whether a mock was called with kwargs, but also some other params
    we don't care about."""
    for call in mock.mock_calls:
        check_call_contains_kwargs(call, params)


class TestTSNEParameterFlow(unittest.TestCase):
    """est that the optimization parameters get properly propagated."""

    grad_descent_params = {
        "negative_gradient_method": [kl_divergence_bh, kl_divergence_fft],
        "learning_rate": [1, 10, 100],
        "theta": [0.2, 0.5, 0.8],
        "n_interpolation_points": [3, 5],
        "min_num_intervals": [10, 20, 30],
        "ints_in_interval": [1, 2, 5],
        "n_jobs": [1, 2, 4],
        "callbacks": [None, [lambda *args, **kwargs: ...]],
        "callbacks_every_iters": [25, 50],
    }

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    @check_params({**grad_descent_params, **{
        "early_exaggeration_iter": [50, 100],
        "early_exaggeration": [4, 12],
        "initial_momentum": [0.2, 0.5, 0.8],
        "n_iter": [50, 100],
        "exaggeration": [None, 2],
        "final_momentum": [0.2, 0.5, 0.8],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_constructor(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Early exaggeration training loop
        if param_name in ("early_exaggeration_iter", "early_exaggeration", "initial_momentum"):
            call_idx = 0
        # Main training loop
        elif param_name in ("n_iter", "exaggeration", "final_momentum"):
            call_idx = 1
        # If general parameter, should be applied to every call
        else:
            call_idx = 0

        TSNE(**{param_name: param_value}).fit(self.x)

        self.assertEqual(2, gradient_descent.call_count)
        check_call_contains_kwargs(
            gradient_descent.mock_calls[call_idx],
            {param_name: param_value},
        )

    @check_params({**grad_descent_params, **{
        "n_iter": [50, 100, 150],
        "exaggeration": [None, 2, 5],
        "momentum": [0.2, 0.5, 0.8],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # `optimize` requires us to specify the `n_iter`
        params = {"n_iter": 50, param_name: param_value}

        tsne = TSNE()
        embedding = tsne.prepare_initial(self.x)
        embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)

    @check_params({
        "n_iter": [50, 100],
        "momentum": [0.2, 0.5, 0.8],
    })
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_embedding_transform(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        embedding.transform(self.x_test, **{param_name: param_value})

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(
            gradient_descent.mock_calls[0],
            {param_name: param_value},
        )

    @check_params({**grad_descent_params, **{
        "n_iter": [50, 100, 150],
        "exaggeration": [None, 2, 5],
        "momentum": [0.2, 0.5, 0.8],
    }})
    @patch("openTSNE.tsne.gradient_descent.__call__")
    def test_partial_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        # `optimize` requires us to specify the `n_iter`
        params = {"n_iter": 50, param_name: param_value}

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)

    @check_params({"metric": set(NNDescent.VALID_METRICS) - {"mahalanobis"}})
    @patch("openTSNE.pynndescent.NNDescent")
    def test_nndescent_distances(self, param_name, metric, nndescent: MagicMock):
        """Distance metrics should be properly passed down to NN descent"""
        assert param_name == "metric"
        tsne = TSNE(metric=metric, neighbors="approx")

        # We don't care about what happens later, just that the NN method is
        # properly called
        nndescent.side_effect = InterruptedError()
        try:
            # Haversine distance only supports two dimensions
            tsne.prepare_initial(self.x[:, :2])
        except InterruptedError:
            pass

        self.assertEqual(nndescent.call_count, 1)
        check_call_contains_kwargs(nndescent.mock_calls[0], {"metric": metric})

    @patch("openTSNE.pynndescent.NNDescent")
    def test_nndescent_mahalanobis_distance(self, nndescent: MagicMock):
        """Distance metrics and additional params should be correctly passed down to NN descent"""
        metric = "mahalanobis"
        C = np.cov(self.x)
        tsne = TSNE(metric=metric, metric_params={"V": C}, neighbors="approx")

        # We don't care about what happens later, just that the NN method is
        # properly called
        nndescent.side_effect = InterruptedError()
        try:
            tsne.prepare_initial(self.x)
        except InterruptedError:
            pass

        self.assertEqual(nndescent.call_count, 1)
        check_call_contains_kwargs(nndescent.mock_calls[0], {"metric": metric})

    def test_raises_error_on_unrecognized_metric(self):
        """Unknown distance metric should raise error"""
        tsne = TSNE(metric="imaginary", neighbors="exact")
        with self.assertRaises(ValueError):
            tsne.prepare_initial(self.x)

        tsne = TSNE(metric="imaginary", neighbors="approx")
        with self.assertRaises(ValueError):
            tsne.prepare_initial(self.x)


class TestTSNEInplaceOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    def test_embedding_inplace_optimization(self):
        embedding1 = self.tsne.prepare_initial(self.x)

        embedding2 = embedding1.optimize(n_iter=5, inplace=True)
        embedding3 = embedding2.optimize(n_iter=5, inplace=True)

        self.assertIs(embedding1.base, embedding2.base)
        self.assertIs(embedding2.base, embedding3.base)

    def test_embedding_not_inplace_optimization(self):
        embedding1 = self.tsne.prepare_initial(self.x)

        embedding2 = embedding1.optimize(n_iter=5, inplace=False)
        embedding3 = embedding2.optimize(n_iter=5, inplace=False)

        self.assertFalse(embedding1.base is embedding2.base)
        self.assertFalse(embedding2.base is embedding3.base)
        self.assertFalse(embedding1.base is embedding3.base)

    def test_partial_embedding_inplace_optimization(self):
        # Prepare reference embedding
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        partial_embedding1 = embedding.prepare_partial(self.x_test)
        partial_embedding2 = partial_embedding1.optimize(5, inplace=True)
        partial_embedding3 = partial_embedding2.optimize(5, inplace=True)

        self.assertIs(partial_embedding1.base, partial_embedding2.base)
        self.assertIs(partial_embedding2.base, partial_embedding3.base)

    def test_partial_embedding_not_inplace_optimization(self):
        # Prepare reference embedding
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        partial_embedding1 = embedding.prepare_partial(self.x_test)
        partial_embedding2 = partial_embedding1.optimize(5, inplace=False)
        partial_embedding3 = partial_embedding2.optimize(5, inplace=False)

        self.assertFalse(partial_embedding1.base is partial_embedding2.base)
        self.assertFalse(partial_embedding2.base is partial_embedding3.base)
        self.assertFalse(partial_embedding1.base is partial_embedding3.base)


class TestTSNECallbackParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    def test_can_pass_callbacks_to_tsne_object(self):
        callback = MagicMock()
        callback2 = MagicMock()
        # We don't want individual callbacks to be iterable
        del callback.__iter__
        del callback2.__iter__

        # Should be able to pass a single callback
        TSNE(callbacks=callback, callbacks_every_iters=1,
             early_exaggeration_iter=0, n_iter=1).fit(self.x)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        TSNE(callbacks=[callback], callbacks_every_iters=1,
             early_exaggeration_iter=0, n_iter=1).fit(self.x)
        self.assertEqual(callback.call_count, 1)

        # Should be able to change the callback on the object
        callback.reset_mock()
        tsne = TSNE(callbacks=callback, callbacks_every_iters=1,
                    early_exaggeration_iter=0, n_iter=1)
        tsne.callbacks = callback2
        tsne.fit(self.x)
        callback.assert_not_called()
        self.assertEqual(callback2.call_count, 1)

    def test_can_pass_callbacks_to_embedding_optimize(self):
        embedding = self.tsne.prepare_initial(self.x)

        # We don't the callback to be iterable
        callback = MagicMock()
        del callback.__iter__

        # Should be able to pass a single callback
        embedding.optimize(1, callbacks=callback, callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        embedding.optimize(1, callbacks=[callback], callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

    def test_can_pass_callbacks_to_partial_embedding_optimize(self):
        embedding = self.tsne.prepare_initial(self.x)

        # We don't the callback to be iterable
        callback = MagicMock()
        del callback.__iter__

        # Should be able to pass a single callback
        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(1, callbacks=callback, callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        partial_embedding.optimize(1, callbacks=[callback], callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)


class TSNEInitialization(unittest.TestCase):
    transform_initializations = ["random", "median", "weighted"]

    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(100, 50, (25, 4))
        cls.iris = datasets.load_iris()["data"]

    def test_low_variance(self):
        """Low variance of the initial embedding is very important for the
        convergence of tSNE."""
        # Cycle through various initializations
        initializations = ["random", "pca"]
        allowed = 1e-3

        for init in initializations:
            tsne = TSNE(initialization=init, perplexity=2)
            embedding = tsne.prepare_initial(self.x)
            np.testing.assert_array_less(np.var(embedding, axis=0), allowed,
                                         "using the `%s` initialization" % init)

    def test_mismatching_embedding_dimensions_simple_api(self):
        # Fit
        tsne = TSNE(n_components=2, initialization=self.x[:10, :2])
        with self.assertRaises(ValueError, msg="fit::incorrect number of points"):
            tsne.fit(self.x[:25])

        with self.assertRaises(ValueError, msg="fit::incorrect number of dimensions"):
            TSNE(n_components=2, initialization=self.x[:10, :4])

        # Transform
        tsne = TSNE(n_components=2, initialization="random")
        embedding = tsne.fit(self.x)
        with self.assertRaises(ValueError, msg="transform::incorrect number of points"):
            embedding.transform(X=self.x[:5], initialization=self.x[:10, :2])

        with self.assertRaises(ValueError, msg="transform::incorrect number of dimensions"):
            embedding.transform(X=self.x, initialization=self.x[:, :4])

    def test_same_unoptimized_initializations_for_transform(self):
        """Initializations should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=50,
            n_iter=50,
            neighbors="exact",
            negative_gradient_method="bh",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.prepare_partial(x_test, initialization=init)
            new_embedding_2 = embedding.prepare_partial(x_test, initialization=init)

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)

    def test_same_bh_optimized_median_initializations_for_transform(self):
        """Transform with Barnes-Hut optimization should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=10,
            n_iter=10,
            neighbors="exact",
            negative_gradient_method="bh",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.transform(
                x_test, initialization=init, n_iter=10
            )
            new_embedding_2 = embedding.transform(
                x_test, initialization=init, n_iter=10
            )

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)

    def test_same_fft_optimized_median_initializations_for_transform(self):
        """Transform with interpolation based optimization should be deterministic."""
        x_train, x_test = train_test_split(self.iris, test_size=0.33, random_state=42)

        embedding = openTSNE.TSNE(
            early_exaggeration_iter=10,
            n_iter=10,
            neighbors="exact",
            negative_gradient_method="fft",
            random_state=42,
        ).fit(x_train)

        for init in self.transform_initializations:
            new_embedding_1 = embedding.transform(
                x_test, initialization=init, n_iter=10, learning_rate=10
            )
            new_embedding_2 = embedding.transform(
                x_test, initialization=init, n_iter=10, learning_rate=10
            )

            np.testing.assert_equal(new_embedding_1, new_embedding_2, init)


class TestRandomState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(10000, 50, (25, 4))
        cls.x_test = np.random.normal(100, 50, (25, 4))

    def test_same_results_on_fixed_random_state_random_init(self):
        """Results should be exactly the same if we provide a random state."""
        tsne1 = TSNE(random_state=1, initialization="random")
        embedding1 = tsne1.fit(self.x)

        tsne2 = TSNE(random_state=1, initialization="random")
        embedding2 = tsne2.fit(self.x)

        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Same random state produced different initial embeddings",
        )

    def test_same_results_on_fixed_random_state_pca_init(self):
        """Results should be exactly the same if we provide a random state."""
        tsne1 = TSNE(random_state=1, initialization="pca")
        embedding1 = tsne1.fit(self.x)

        tsne2 = TSNE(random_state=1, initialization="pca")
        embedding2 = tsne2.fit(self.x)

        np.testing.assert_array_equal(
            embedding1,
            embedding2,
            "Same random state produced different initial embeddings",
        )

    def test_same_partial_embedding_on_fixed_random_state(self):
        tsne = TSNE(random_state=1, initialization="random")
        embedding = tsne.fit(self.x)

        partial1 = embedding.prepare_partial(self.x_test, initialization="random")
        partial2 = embedding.prepare_partial(self.x_test, initialization="random")

        np.testing.assert_array_equal(
            partial1,
            partial2,
            "Same random state produced different partial embeddings",
        )

    @patch("openTSNE.initialization.random", wraps=openTSNE.initialization.random)
    @patch("openTSNE.nearest_neighbors.BallTree", wraps=openTSNE.nearest_neighbors.BallTree)
    def test_random_state_parameter_is_propagated_random_init_exact(self, init, neighbors):
        random_state = 1

        tsne = openTSNE.TSNE(
            neighbors="exact",
            initialization="random",
            random_state=random_state,
        )
        tsne.prepare_initial(self.x)

        # Verify that `random_state` was passed
        init.assert_called_once()
        check_mock_called_with_kwargs(init, {"random_state": random_state})
        neighbors.assert_called_once()
        check_mock_called_with_kwargs(neighbors, {"random_state": random_state})

    @patch("openTSNE.initialization.pca", wraps=openTSNE.initialization.pca)
    @patch("openTSNE.nearest_neighbors.NNDescent", wraps=openTSNE.nearest_neighbors.NNDescent)
    def test_random_state_parameter_is_propagated_pca_init_approx(self, init, neighbors):
        random_state = 1

        tsne = openTSNE.TSNE(
            neighbors="approx",
            initialization="pca",
            random_state=random_state,
        )
        tsne.prepare_initial(self.x)

        # Verify that `random_state` was passed
        init.assert_called_once()
        check_mock_called_with_kwargs(init, {"random_state": random_state})
        neighbors.assert_called_once()
        check_mock_called_with_kwargs(neighbors, {"random_state": random_state})


class TestDefaultParameterSettings(unittest.TestCase):
    def test_default_params_simple_vs_complex_flow(self):
        # Relevant affinity parameters are passed to the affinity object
        mismatching = get_mismatching_default_values(
            openTSNE.TSNE,
            PerplexityBasedNN,
            {"neighbors": "method"},
        )
        self.assertEqual(mismatching, [])

        assert len(
            get_shared_parameters(openTSNE.TSNE, openTSNE.tsne.gradient_descent.__call__)
        ) > 0, \
            "`TSNE` and `gradient_descent` have no shared parameters. Have you " \
            "changed the signature or usage?"

        # The relevant gradient descent parameters are passed down directly to
        # `gradient_descent`
        mismatching = get_mismatching_default_values(
            openTSNE.TSNE,
            openTSNE.tsne.gradient_descent.__call__,
        )
        mismatching = list(filter(lambda x: x[0] not in ("n_iter",), mismatching))
        self.assertEqual(mismatching, [])


def get_shared_parameters(f1, f2):
    """Get the names of shared parameters from two function signatures."""
    params1 = inspect.signature(f1).parameters
    params2 = inspect.signature(f2).parameters

    return set(params1.keys()) & set(params2.keys())


def get_mismatching_default_values(f1, f2, mapping=None):
    """Check that two functions have the same default values for shared parameters."""
    # Additional mappings from f1 parameters to f2 parameters may be provided
    if mapping is None:
        mapping = {}

    params1 = inspect.signature(f1).parameters
    params2 = inspect.signature(f2).parameters

    mismatch = []
    for f1_param_name in params1:
        # If the param is named differently in f2, rename
        f2_param_name = mapping[f1_param_name] if f1_param_name in mapping else f1_param_name

        # If the parameter does not appear in the signature of f2, there"s
        # nothing to do
        if f2_param_name not in params2:
            continue

        val1 = params1[f1_param_name].default
        val2 = params2[f2_param_name].default

        if val1 != val2:
            mismatch.append((f1_param_name, val1, f2_param_name, val2))

    return mismatch


class TestGradientDescentOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = TSNE()
        random_state = np.random.RandomState(42)
        cls.x = random_state.randn(100, 4)
        cls.x_test = random_state.randn(25, 4)

    def test_optimizer_being_passed_to_subsequent_embeddings(self):
        embedding = self.tsne.prepare_initial(self.x)

        self.assertIsNone(embedding.optimizer.gains,
                          "Optimizer should be initialized with no gains")

        # Check the switch from no gains to some gains
        embedding1 = embedding.optimize(10)
        self.assertIsNone(
            embedding.optimizer.gains,
            "Gains changed on initial optimizer even though we did not do "
            "inplace optimization.")
        self.assertIsNotNone(
            embedding1.optimizer.gains,
            "Gains were not properly set in new embedding.")
        self.assertIsNot(
            embedding.optimizer, embedding1.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.")

        # Check switch from existing gains to new gains
        embedding2 = embedding1.optimize(10)
        self.assertIsNot(
            embedding1.optimizer, embedding2.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.")
        self.assertFalse(
            np.allclose(embedding1.optimizer.gains, embedding2.optimizer.gains),
            "The gains in the new embedding did not change at all from the old "
            "embedding."
        )

    def test_optimizer_being_passed_to_partial_embeddings(self):
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)

        # Partial embeddings get their own optimizer instance
        partial = embedding.prepare_partial(self.x_test)
        self.assertIsNot(
            embedding.optimizer, partial.optimizer,
            "Embedding and partial embedding optimizers are the same instance.")
        self.assertIsNone(
            partial.optimizer.gains,
            "Partial embedding was not initialized with no gains")

        # Check the switch from no gains to some gains
        partial1 = partial.optimize(10)
        self.assertIsNone(
            partial.optimizer.gains,
            "Gains on initial optimizer changed even though we did not do "
            "inplace optimization.")
        self.assertIsNotNone(
            partial1.optimizer.gains,
            "Gains were not properly set in new partial embedding.")

        # Check switch from existing gains to new gains
        partial2 = partial1.optimize(10)
        self.assertIsNot(
            partial1.optimizer, partial2.optimizer,
            "The embedding and new embedding optimizer are the same instance "
            "even we did not do inplace optimization.")
        self.assertFalse(
            np.allclose(partial1.optimizer.gains, partial2.optimizer.gains),
            "The gains in the new embedding did not change at all from the old "
            "embedding."
        )

    def test_embedding_optimizer_inplace(self):
        embedding0 = self.tsne.prepare_initial(self.x)

        # Assign only so the references are clear
        embedding1 = embedding0.optimize(10, inplace=True)
        embedding2 = embedding1.optimize(10, inplace=True)

        self.assertIs(embedding0.optimizer, embedding1.optimizer)
        self.assertIs(embedding1.optimizer, embedding2.optimizer)

    def test_partial_embedding_optimizer_inplace(self):
        embedding = self.tsne.prepare_initial(self.x)
        embedding.optimize(10, inplace=True)
        partial0 = embedding.prepare_partial(self.x_test)

        # Assign only so the references are clear
        partial1 = partial0.optimize(10, inplace=True)
        partial2 = partial1.optimize(10, inplace=True)

        self.assertIs(partial0.optimizer, partial1.optimizer)
        self.assertIs(partial1.optimizer, partial2.optimizer)


class TestAffinityIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # It would be nice if the initial data were not nicely behaved to test
        # for low variance
        cls.x = np.random.normal(100, 50, (25, 4))
        cls.x_test = np.random.normal(100, 50, (25, 4))

    def test_transform_with_standard_affinity(self):
        init = openTSNE.initialization.random(self.x)
        aff = openTSNE.affinity.PerplexityBasedNN(self.x, 5, method="exact")
        embedding = openTSNE.TSNEEmbedding(init, aff, negative_gradient_method="bh")
        embedding.optimize(100, inplace=True)

        # This should not raise an error
        embedding.transform(self.x_test)

    def test_transform_with_nonstandard_affinity(self):
        """Should raise an informative error when a non-standard affinity is used
        with `transform`."""
        init = openTSNE.initialization.random(self.x)
        aff = openTSNE.affinity.Multiscale(self.x, [2, 5], method="exact")
        embedding = openTSNE.TSNEEmbedding(init, aff, negative_gradient_method="bh")
        embedding.optimize(100, inplace=True)

        with self.assertRaises(TypeError):
            embedding.transform(self.x_test)

import unittest
from functools import wraps
from typing import Callable, Any, Tuple
from unittest.mock import patch, MagicMock

import numpy as np

from fastTSNE.tsne import TSNE, kl_divergence_bh, kl_divergence_fft

np.random.seed(42)


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


def check_call_contains_kwargs(call: Tuple, params: dict) -> None:
    """Check whether a `call` object was called with some params, but also some
    others we don't care about"""
    name, args, kwargs = call
    for key in params:
        expected_value = params[key]
        actual_value = kwargs.get(key, None)
        if expected_value != actual_value:
            raise AssertionError(
                'Mock not called with `%s=%s`. Called with `%s`' %
                (key, expected_value, actual_value)
            )


def check_mock_called_with_kwargs(mock: MagicMock, params: dict) -> None:
    """Check whether a mock was called with kwargs, but also some other params
    we don't care about."""
    for call in mock.mock_calls:
        check_call_contains_kwargs(call, params)


class TestTSNEParameterFlow(unittest.TestCase):
    """Test that the optimization parameters get properly propagated."""

    grad_descent_params = {
        'negative_gradient_method': [kl_divergence_bh, kl_divergence_fft],
        'learning_rate': [1, 10, 100],
        'theta': [0.2, 0.5, 0.8],
        'n_interpolation_points': [3, 5],
        'min_num_intervals': [10, 20, 30],
        'ints_in_interval': [1, 2, 5],
        'n_jobs': [1, 2, 4],
        'callbacks': [None, [lambda *args, **kwargs: ...]],
        'callbacks_every_iters': [25, 50],
    }

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

    @check_params({**grad_descent_params, **{
        'early_exaggeration_iter': [50, 100],
        'early_exaggeration': [4, 12],
        'initial_momentum': [0.2, 0.5, 0.8],
        'n_iter': [50, 100],
        'final_momentum': [0.2, 0.5, 0.8],
        'late_exaggeration_iter': [50, 100],
        'late_exaggeration': [None, 2],
    }})
    @patch('fastTSNE.tsne.gradient_descent')
    def test_constructor(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Early exaggeration training loop
        if param_name == 'early_exaggeration_iter':
            check_param_name = 'n_iter'
            call_idx = 0
        elif param_name == 'early_exaggeration':
            check_param_name = 'exaggeration'
            call_idx = 0
        elif param_name == 'initial_momentum':
            check_param_name = 'momentum'
            call_idx = 0
        # Main training loop
        elif param_name == 'n_iter':
            check_param_name = param_name
            call_idx = 1
        elif param_name == 'final_momentum':
            check_param_name = 'momentum'
            call_idx = 1
        # Early exaggeration training loop
        elif param_name == 'late_exaggeration_iter':
            check_param_name = 'n_iter'
            call_idx = 2
        elif param_name == 'late_exaggeration':
            check_param_name = 'exaggeration'
            call_idx = 2

        # If general parameter, should be applied to every call
        else:
            check_param_name = param_name
            call_idx = 0

        TSNE(**{param_name: param_value}).fit(self.x)

        self.assertEqual(3, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[call_idx],
                                   {check_param_name: param_value})

    @check_params({**grad_descent_params, **{
        'n_iter': [50, 100, 150],
        'exaggeration': [None, 2, 5],
        'momentum': [0.2, 0.5, 0.8],
    }})
    @patch('fastTSNE.tsne.gradient_descent')
    def test_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # `optimize` requires us to specify the `n_iter`
        params = {'n_iter': 50, param_name: param_value}

        tsne = TSNE()
        embedding = tsne.prepare_initial(self.x)
        embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)

    @check_params({**grad_descent_params, **{
        'early_exaggeration_iter': [50, 100],
        'early_exaggeration': [4, 12],
        'initial_momentum': [0.2, 0.5, 0.8],
        'n_iter': [50, 100],
        'final_momentum': [0.2, 0.5, 0.8],
    }})
    @patch('fastTSNE.tsne.gradient_descent')
    def test_embedding_transform(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        embedding.transform(self.x_test, **{param_name: param_value})

        # Early exaggeration training loop
        if param_name == 'early_exaggeration_iter':
            check_param_name = 'n_iter'
            call_idx = 0
        elif param_name == 'early_exaggeration':
            check_param_name = 'exaggeration'
            call_idx = 0
        elif param_name == 'initial_momentum':
            check_param_name = 'momentum'
            call_idx = 0
        # Main training loop
        elif param_name == 'n_iter':
            check_param_name = param_name
            call_idx = 1
        elif param_name == 'final_momentum':
            check_param_name = 'momentum'
            call_idx = 1

        # If general parameter, should be applied to every call
        else:
            check_param_name = param_name
            call_idx = 0

        self.assertEqual(2, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[call_idx],
                                   {check_param_name: param_value})

    @check_params({**grad_descent_params, **{
        'n_iter': [50, 100, 150],
        'exaggeration': [None, 2, 5],
        'momentum': [0.2, 0.5, 0.8],
    }})
    @patch('fastTSNE.tsne.gradient_descent')
    def test_partial_embedding_optimize(self, param_name, param_value, gradient_descent):
        # type: (str, Any, MagicMock) -> None
        # Make sure mock still conforms to signature
        gradient_descent.return_value = (1, MagicMock())

        # Perform initial embedding - this is tested above
        tsne = TSNE()
        embedding = tsne.fit(self.x)
        gradient_descent.reset_mock()

        # `optimize` requires us to specify the `n_iter`
        params = {'n_iter': 50, param_name: param_value}

        partial_embedding = embedding.prepare_partial(self.x_test)
        partial_embedding.optimize(**params, inplace=True)

        self.assertEqual(1, gradient_descent.call_count)
        check_call_contains_kwargs(gradient_descent.mock_calls[0], params)


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

    def test_can_pass_callbacks_to_embedding_transform(self):
        embedding = self.tsne.prepare_initial(self.x)

        # We don't the callback to be iterable
        callback = MagicMock()
        del callback.__iter__

        # Should be able to pass a single callback
        embedding.transform(self.x_test, early_exaggeration_iter=0, n_iter=1,
                            callbacks=callback, callbacks_every_iters=1)
        self.assertEqual(callback.call_count, 1)

        # Should be able to pass a list callbacks
        callback.reset_mock()
        embedding.transform(self.x_test, early_exaggeration_iter=0, n_iter=1,
                            callbacks=[callback], callbacks_every_iters=1)
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
    @classmethod
    def setUpClass(cls):
        cls.x = np.random.randn(100, 4)
        cls.x_test = np.random.randn(25, 4)

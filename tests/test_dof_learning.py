"""Tests for the learnable degrees-of-freedom feature."""
import logging
import unittest
import warnings
from functools import partial

import numpy as np
from sklearn import datasets

import openTSNE
from openTSNE import affinity
from openTSNE.tsne import (
    GradientResult,
    IterationState,
    kl_divergence_bh,
    kl_divergence_fft,
)

np.random.seed(42)
affinity.log.setLevel(logging.ERROR)

TSNE_BH = partial(
    openTSNE.TSNE, neighbors="exact", negative_gradient_method="bh", random_state=42
)
TSNE_FFT = partial(
    openTSNE.TSNE, neighbors="exact", negative_gradient_method="fft", random_state=42
)


class _ObjectiveFnMixin:
    """Build a small (embedding, P) pair the objective functions can chew on."""

    @classmethod
    def setUpClass(cls):
        x = datasets.load_iris()["data"]
        aff = affinity.PerplexityBasedNN(x, perplexity=30, random_state=42)
        cls.P = aff.P
        cls.embedding = np.random.RandomState(0).randn(x.shape[0], 2) * 1e-4
        cls.bh_params = {"theta": 0.5}
        cls.fft_params = {
            "n_interpolation_points": 3,
            "min_num_intervals": 10,
            "ints_in_interval": 1,
        }


class TestKLDivergenceBHReturnType(_ObjectiveFnMixin, unittest.TestCase):
    def test_returns_gradient_result(self):
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            should_eval_error=True,
        )
        self.assertIsInstance(result, GradientResult)
        self.assertEqual(result.gradient.shape, self.embedding.shape)
        self.assertTrue(np.isfinite(result.error))

    def test_compute_dof_grad_false_yields_zero(self):
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=False,
        )
        self.assertEqual(result.dof_grad, 0.0)

    def test_compute_dof_grad_true_yields_nonzero(self):
        # For a non-trivial random embedding far from optimum, the dof gradient
        # should be measurably away from 0.
        result = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=True,
        )
        self.assertNotEqual(result.dof_grad, 0.0)
        self.assertTrue(np.isfinite(result.dof_grad))

    def test_gradient_unaffected_by_compute_dof_grad(self):
        # Toggling the flag must not change the embedding gradient itself.
        r1 = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=False,
        )
        r2 = kl_divergence_bh(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            bh_params=self.bh_params,
            compute_dof_grad=True,
        )
        np.testing.assert_allclose(r1.gradient, r2.gradient)

    def test_dof_grad_matches_finite_difference(self):
        """Numerical correctness check: dof_grad must match a central-difference
        estimate of dKL/d(dof) computed at the same embedding.

        The analytical and finite-difference paths both use the same Barnes-Hut
        approximation, so the BH error cancels out and only the finite-difference
        truncation error matters. P is normalized (sum_P = 1, no exaggeration),
        which is the regime where the implementation's asymmetric normalization
        is consistent with the true gradient.
        """
        rng = np.random.RandomState(7)
        # Use a non-tiny embedding so dof_grad has meaningful magnitude.
        embedding = rng.randn(self.embedding.shape[0], 2) * 0.5
        dof = 1.0
        h = 1e-4

        analytical = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof,
            bh_params=self.bh_params,
            should_eval_error=True,
            compute_dof_grad=True,
        ).dof_grad

        kl_plus = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof + h,
            bh_params=self.bh_params,
            should_eval_error=True,
        ).error
        kl_minus = kl_divergence_bh(
            embedding.copy(),
            self.P,
            dof=dof - h,
            bh_params=self.bh_params,
            should_eval_error=True,
        ).error
        finite_diff = (kl_plus - kl_minus) / (2 * h)

        # Sanity: gradient should be non-trivial; otherwise the test is vacuous.
        self.assertGreater(abs(analytical), 1e-3)

        rel_err = abs(analytical - finite_diff) / abs(analytical)
        self.assertLess(
            rel_err,
            5e-3,
            f"dof_grad={analytical:.6g}, finite_diff={finite_diff:.6g}, "
            f"rel_err={rel_err:.3g}",
        )


class TestKLDivergenceFFTReturnType(_ObjectiveFnMixin, unittest.TestCase):
    def test_returns_gradient_result(self):
        result = kl_divergence_fft(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            fft_params=self.fft_params,
            should_eval_error=True,
        )
        self.assertIsInstance(result, GradientResult)
        self.assertEqual(result.gradient.shape, self.embedding.shape)

    def test_compute_dof_grad_false_yields_zero(self):
        result = kl_divergence_fft(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            fft_params=self.fft_params,
            compute_dof_grad=False,
        )
        self.assertEqual(result.dof_grad, 0.0)

    def test_compute_dof_grad_true_yields_nonzero(self):
        # Both the dof=1 fast path and the general path must produce a real
        # gradient, since learning starts at dof=1 by default.
        for dof in (1.0, 2.5):
            with self.subTest(dof=dof):
                result = kl_divergence_fft(
                    self.embedding.copy(),
                    self.P,
                    dof=dof,
                    fft_params=self.fft_params,
                    compute_dof_grad=True,
                )
                self.assertNotEqual(result.dof_grad, 0.0)
                self.assertTrue(np.isfinite(result.dof_grad))

    def test_gradient_unaffected_by_compute_dof_grad(self):
        # Toggling the flag must not change the embedding gradient itself.
        r1 = kl_divergence_fft(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            fft_params=self.fft_params,
            compute_dof_grad=False,
        )
        r2 = kl_divergence_fft(
            self.embedding.copy(),
            self.P,
            dof=1.0,
            fft_params=self.fft_params,
            compute_dof_grad=True,
        )
        np.testing.assert_allclose(r1.gradient, r2.gradient)

    def test_dof_grad_matches_finite_difference(self):
        """The analytic dof_grad must match a central-difference estimate of
        dKL/d(dof) computed through the same FFT objective, so the
        interpolation error cancels and only the truncation error remains."""
        rng = np.random.RandomState(7)
        embedding = rng.randn(self.embedding.shape[0], 2) * 0.5
        h = 1e-4

        for dof in (0.5, 1.0, 2.5):
            with self.subTest(dof=dof):
                analytical = kl_divergence_fft(
                    embedding.copy(),
                    self.P,
                    dof=dof,
                    fft_params=self.fft_params,
                    should_eval_error=True,
                    compute_dof_grad=True,
                ).dof_grad
                kl_plus = kl_divergence_fft(
                    embedding.copy(),
                    self.P,
                    dof=dof + h,
                    fft_params=self.fft_params,
                    should_eval_error=True,
                ).error
                kl_minus = kl_divergence_fft(
                    embedding.copy(),
                    self.P,
                    dof=dof - h,
                    fft_params=self.fft_params,
                    should_eval_error=True,
                ).error
                finite_diff = (kl_plus - kl_minus) / (2 * h)

                self.assertGreater(abs(analytical), 1e-3)
                rel_err = abs(analytical - finite_diff) / abs(analytical)
                self.assertLess(
                    rel_err,
                    1e-4,
                    f"dof_grad={analytical:.6g}, finite_diff={finite_diff:.6g}, "
                    f"rel_err={rel_err:.3g}",
                )

    def test_dof_grad_matches_exact_computation(self):
        """The FFT dof_grad must match the exact O(N^2) gradient to
        interpolation tolerance across heavy-, standard-, and light-tailed
        regimes."""
        rng = np.random.RandomState(7)
        embedding = rng.randn(self.embedding.shape[0], 2) * 0.5

        def exact_dof_grad(emb, P, dof):
            d = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
            k = (1 + d / dof) ** -dof
            g = np.log1p(d / dof) - d / (dof + d + np.finfo(float).tiny)
            np.fill_diagonal(k, 0.0)
            np.fill_diagonal(g, 0.0)
            a_pos = (P.toarray() * g).sum()
            a_neg = (k * g).sum() / k.sum()
            return a_pos - a_neg

        for dof in (0.3, 1.0, 5.0):
            with self.subTest(dof=dof):
                exact = exact_dof_grad(embedding, self.P, dof)
                fft = kl_divergence_fft(
                    embedding.copy(),
                    self.P,
                    dof=dof,
                    fft_params=self.fft_params,
                    compute_dof_grad=True,
                ).dof_grad
                rel_err = abs(fft - exact) / abs(exact)
                self.assertLess(
                    rel_err,
                    1e-4,
                    f"fft={fft:.10g}, exact={exact:.10g}, rel_err={rel_err:.3g}",
                )

    def test_dof_grad_matches_exact_computation_1d(self):
        rng = np.random.RandomState(7)
        embedding = rng.randn(self.embedding.shape[0], 1) * 0.5

        d = (embedding[:, None, 0] - embedding[None, :, 0]) ** 2
        dof = 2.5
        k = (1 + d / dof) ** -dof
        g = np.log1p(d / dof) - d / (dof + d + np.finfo(float).tiny)
        np.fill_diagonal(k, 0.0)
        np.fill_diagonal(g, 0.0)
        exact = (self.P.toarray() * g).sum() - (k * g).sum() / k.sum()

        fft = kl_divergence_fft(
            embedding.copy(),
            self.P,
            dof=dof,
            fft_params=self.fft_params,
            compute_dof_grad=True,
        ).dof_grad
        rel_err = abs(fft - exact) / abs(exact)
        self.assertLess(rel_err, 1e-3, f"fft={fft:.10g}, exact={exact:.10g}")


class TestOptimizerReturnShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_fit_returns_embedding_with_kl(self):
        emb = TSNE_BH(early_exaggeration_iter=5, n_iter=5).fit(self.x)
        self.assertTrue(hasattr(emb, "kl_divergence_"))
        self.assertTrue(np.isfinite(emb.kl_divergence_))

    def test_kl_divergence_deprecated_alias(self):
        emb = TSNE_BH(early_exaggeration_iter=5, n_iter=5).fit(self.x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = emb.kl_divergence
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "Reading `kl_divergence` should emit a DeprecationWarning.",
        )
        self.assertEqual(value, emb.kl_divergence_)


class TestDofAutoLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_bh_auto_actually_moves_dof(self):
        # With dof="auto" on the BH path, dof should change over iterations.
        history = []
        TSNE_BH(
            dof="auto",
            initial_dof=1.0,
            early_exaggeration_iter=0,
            n_iter=30,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        dofs = [s.dof for s in history]
        self.assertGreater(len(dofs), 0)
        self.assertNotEqual(dofs[0], dofs[-1])

    def test_auto_dof_reaches_lower_kl_than_fixed_dof(self):
        # `dof="auto"` should always achieve lower KL than `dof=1`
        fixed = TSNE_BH(dof=1.0, n_iter=100).fit(self.x)
        auto = TSNE_BH(dof="auto", initial_dof=1.0, n_iter=100).fit(self.x)
        self.assertLess(auto.kl_divergence_, fixed.kl_divergence_)

    def test_default_dof_no_initial_dof(self):
        # Path 1: dof=1 (default), initial_dof=None. Dof should stay at 1
        # throughout, no warnings about initial_dof.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(all(s.dof == 1.0 for s in history))
        self.assertFalse(
            any("initial_dof" in str(w.message) for w in caught),
            f"Did not expect initial_dof warning, got: {[str(w.message) for w in caught]}",
        )

    def test_fixed_dof_with_initial_dof_warns(self):
        # Path 2: dof=1 (fixed), initial_dof=5. initial_dof must be ignored
        # and a warning must be emitted.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                initial_dof=5.0,
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(
            any("initial_dof" in str(w.message) for w in caught),
            f"Expected initial_dof warning, got: {[str(w.message) for w in caught]}",
        )
        self.assertTrue(all(s.dof == 1.0 for s in history))

    def test_nondefault_fixed_dof_with_initial_dof_warns(self):
        # Path 2b: dof=2.5 (non-default fixed), initial_dof=5. Same warning;
        # dof stays at 2.5.
        history = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                dof=2.5,
                initial_dof=5.0,
                early_exaggeration_iter=0,
                n_iter=10,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(
            any("initial_dof" in str(w.message) for w in caught),
            f"Expected initial_dof warning, got: {[str(w.message) for w in caught]}",
        )
        self.assertTrue(all(s.dof == 2.5 for s in history))

    def test_auto_dof_no_initial_dof_starts_from_one(self):
        # Path 3: dof="auto", initial_dof=None. Starting dof must default to 1.
        history = []
        TSNE_BH(
            dof="auto",
            early_exaggeration_iter=0,
            n_iter=2,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertGreater(len(history), 0)
        # First iteration's `dof` is the starting value before any update.
        self.assertEqual(history[0].dof, 1.0)

    def test_auto_dof_with_initial_dof_starts_from_initial(self):
        # Path 4: dof="auto", initial_dof=5. Starting dof must be 5.
        history = []
        TSNE_BH(
            dof="auto",
            initial_dof=5.0,
            early_exaggeration_iter=0,
            n_iter=2,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0].dof, 5.0)

    def test_auto_dof_rejects_invalid_initial_dof(self):
        for bad in (0, -1.0, np.nan, np.inf, "5"):
            with self.subTest(initial_dof=bad):
                with self.assertRaisesRegex(ValueError, "initial_dof"):
                    TSNE_BH(
                        dof="auto",
                        initial_dof=bad,
                        early_exaggeration_iter=0,
                        n_iter=1,
                    ).fit(self.x)

    def test_fixed_then_auto_resumes_from_fixed(self):
        # Switching from a fixed dof to dof="auto" must pick up at the fixed
        # value, so the user can use a fixed-dof phase as a warm start for
        # subsequent dof learning.
        embedding = TSNE_BH(
            dof=5.0, early_exaggeration_iter=0, n_iter=10,
        ).fit(self.x)
        self.assertEqual(embedding.dof_, 5.0)

        history = []
        embedding.optimize(
            n_iter=2,
            dof="auto",
            inplace=True,
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        self.assertEqual(history[0].dof, 5.0)

    def test_auto_then_fixed_overrides_dof(self):
        # Reverse direction: after learning dof, the user pins it. The fixed
        # value must take effect immediately (visible to the very first
        # iteration) and be reflected on the embedding afterwards.
        embedding = TSNE_BH(
            dof="auto", early_exaggeration_iter=0, n_iter=10,
        ).fit(self.x)
        learned = embedding.dof_
        self.assertIsInstance(learned, float)
        self.assertNotEqual(learned, 7.0)

        history = []
        embedding.optimize(
            n_iter=2,
            dof=7.0,
            inplace=True,
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        self.assertEqual(history[0].dof, 7.0)
        self.assertEqual(embedding.dof_, 7.0)

    def test_embedding_dof_attribute_always_set(self):
        # Every fitted embedding exposes the dof it was optimized with: the
        # fixed value for fixed dof, the learned value for `dof="auto"`. This
        # lets consumers read `embedding.dof_` without branching on the mode.
        fixed_default = TSNE_BH(
            early_exaggeration_iter=0, n_iter=5,
        ).fit(self.x)
        self.assertEqual(fixed_default.dof_, 1.0)

        fixed_nondefault = TSNE_BH(
            dof=2.5, early_exaggeration_iter=0, n_iter=5,
        ).fit(self.x)
        self.assertEqual(fixed_nondefault.dof_, 2.5)

        learned = TSNE_BH(
            dof="auto", early_exaggeration_iter=0, n_iter=5,
        ).fit(self.x)
        self.assertIsInstance(learned.dof_, float)
        self.assertNotEqual(learned.dof_, 1.0)

    def test_dof_attribute_set_at_construction(self):
        # A freshly-constructed (un-optimized) embedding must already expose
        # `dof_`, so callers don't have to branch on whether optimization has
        # run yet: the fixed `dof`, or for `dof="auto"` the value learning
        # will start from. `initial_dof` is not kept as a parameter, so it
        # cannot restart learning on later `optimize` calls.
        from openTSNE import TSNEEmbedding
        from openTSNE import affinity, initialization

        aff = affinity.PerplexityBasedNN(self.x, perplexity=15, random_state=42)
        init = initialization.pca(self.x, random_state=42)

        emb_default = TSNEEmbedding(init, aff, negative_gradient_method="bh")
        self.assertEqual(emb_default.dof_, 1.0)

        emb_fixed = TSNEEmbedding(
            init, aff, negative_gradient_method="bh", dof=2.5,
        )
        self.assertEqual(emb_fixed.dof_, 2.5)

        emb_auto = TSNEEmbedding(
            init, aff, negative_gradient_method="bh", dof="auto",
        )
        self.assertEqual(emb_auto.dof_, 1.0)

        emb_auto_initial = TSNEEmbedding(
            init, aff, negative_gradient_method="bh", dof="auto", initial_dof=3,
        )
        self.assertEqual(emb_auto_initial.dof_, 3.0)
        self.assertNotIn("initial_dof", emb_auto_initial.gradient_descent_params)

        # `prepare_initial` (the path TSNE.fit goes through) must give the
        # same answer.
        prepared = TSNE_BH(dof=2.5).prepare_initial(self.x)
        self.assertEqual(prepared.dof_, 2.5)

    def test_auto_dof_resumes_across_optimize_calls(self):
        # The learned dof must persist on the embedding so a subsequent
        # `optimize()` call resumes from the last learned value rather than
        # restarting from `initial_dof`. Critically, this is the path that
        # `TSNE.fit` itself takes between early-exag and the main optimization.
        embedding = TSNE_BH(
            dof="auto",
            initial_dof=5.0,
            early_exaggeration_iter=0,
            n_iter=20,
        ).fit(self.x)

        learned_dof = embedding.dof_
        self.assertIsNotNone(learned_dof)
        # Sanity: actual learning happened, so the stored value is no longer
        # the initial 5.0 — otherwise the resume assertion below is vacuous.
        self.assertNotEqual(learned_dof, 5.0)

        history = []
        embedding.optimize(
            n_iter=2,
            inplace=True,
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        self.assertEqual(history[0].dof, learned_dof)

    def test_fft_auto_learns_dof(self):
        # The FFT path learns dof for self-embeddings just like BH: no
        # warning, and dof must actually move away from its starting value.
        history = []
        with self.assertNoLogs("openTSNE.tsne", level="WARNING"):
            emb = TSNE_FFT(
                dof="auto",
                initial_dof=1.0,
                early_exaggeration_iter=0,
                n_iter=20,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(history)
        self.assertNotEqual(emb.dof_, 1.0)
        self.assertTrue(all(np.isfinite(s.dof) and s.dof > 0 for s in history))

    def test_fft_auto_learns_dof_1d(self):
        # The 1d FFT estimator is a separate implementation from the 2d one and
        # is only reachable through `n_components=1`, so it needs its own
        # end-to-end check that dof is genuinely learned.
        history = []
        with self.assertNoLogs("openTSNE.tsne", level="WARNING"):
            emb = TSNE_FFT(
                n_components=1,
                dof="auto",
                initial_dof=1.0,
                early_exaggeration_iter=0,
                n_iter=20,
                callbacks=history.append,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertTrue(history)
        self.assertNotEqual(emb.dof_, 1.0)
        self.assertTrue(np.isfinite(emb.dof_) and emb.dof_ > 0)
        self.assertTrue(all(np.isfinite(s.dof) and s.dof > 0 for s in history))

    def test_fft_and_bh_learn_comparable_dof(self):
        # Both approximations descend the same objective, so the learned dof
        # values should land in the same neighborhood.
        emb_bh = TSNE_BH(dof="auto", n_iter=250).fit(self.x)
        emb_fft = TSNE_FFT(dof="auto", n_iter=250).fit(self.x)
        self.assertLess(
            abs(np.log(emb_fft.dof_) - np.log(emb_bh.dof_)),
            np.log(2),
            f"bh dof_={emb_bh.dof_:.4f}, fft dof_={emb_fft.dof_:.4f}",
        )


class TestDofLearningStability(unittest.TestCase):
    """The learnable dof is optimized in log space with an
    exaggeration-normalized gradient and a gain-capped optimizer, so it stays
    positive and well-behaved across exaggeration schedules (see the
    degrees-of-freedom guide)."""

    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_dof_stays_positive_in_log_space(self):
        # Log-space learning can never produce a non-positive dof, even with a
        # tiny initial value and many iterations.
        history = []
        TSNE_BH(
            dof="auto",
            initial_dof=0.05,
            n_iter=100,
            callbacks=history.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertTrue(all(s.dof > 0 for s in history))

    def test_dof_does_not_diverge_when_learned_under_strong_exaggeration(self):
        # `fit` never learns dof under exaggeration, but chained `optimize`
        # calls can, with a warning. The exaggeration-normalized learning rate
        # must still keep dof sane there; without it, dof ran off to the
        # hundreds or thousands at high exaggeration.
        learned = {}
        for ee in (4, 12, 32):
            emb = TSNE_BH(dof="auto").prepare_initial(self.x)
            with self.assertWarnsRegex(UserWarning, "exaggeration"):
                emb.optimize(250, exaggeration=ee, inplace=True)
            emb.optimize(250, inplace=True)
            learned[ee] = emb.dof_
            self.assertLess(emb.dof_, 20.0, f"dof diverged at ee={ee}: {emb.dof_}")
            self.assertGreater(emb.dof_, 0.1)
        # And the final value should be reasonably consistent across schedules.
        vals = np.array(list(learned.values()))
        self.assertLess(vals.std() / vals.mean(), 0.4)


class TestDofEarlyExaggeration(unittest.TestCase):
    """`TSNE.fit` holds dof at 1 through early exaggeration, where the dof
    gradient is not that of the KL divergence, and learns it only in the normal
    phase, starting from `initial_dof`."""

    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def _fit_with_history(self, TSNE_, **kwargs):
        history = []
        params = dict(
            dof="auto",
            early_exaggeration_iter=10,
            n_iter=20,
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        params.update(kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            embedding = TSNE_(**params).fit(self.x)
        return embedding, history, caught

    def test_dof_held_at_one_during_early_exaggeration(self):
        for name, TSNE_ in (("bh", TSNE_BH), ("fft", TSNE_FFT)):
            with self.subTest(method=name):
                embedding, history, _ = self._fit_with_history(TSNE_)
                early = [s for s in history if s.exaggeration == 12]
                normal = [s for s in history if s.exaggeration == 1]
                self.assertEqual(len(early), 10)
                self.assertEqual(len(normal), 20)
                self.assertTrue(all(s.dof == 1 for s in early))
                self.assertTrue(all(s.dof_grad == 0 for s in early))
                self.assertTrue(all(s.dof_grad != 0 for s in normal))
                self.assertNotEqual(embedding.dof_, 1.0)

    def test_learning_starts_from_initial_dof_after_early_exaggeration(self):
        embedding, history, caught = self._fit_with_history(TSNE_BH, initial_dof=2)
        early = [s for s in history if s.exaggeration == 12]
        normal = [s for s in history if s.exaggeration == 1]
        self.assertTrue(all(s.dof == 1 for s in early))
        self.assertEqual(normal[0].dof, 2.0)
        # Neither the pinned phase nor the hand-off warns
        messages = [str(w.message) for w in caught]
        self.assertFalse(
            any("initial_dof" in m or "exaggeration" in m for m in messages),
            messages,
        )

    def test_zero_normal_iterations_records_initial_dof(self):
        # The normal phase sets `dof_` to `initial_dof` even if it runs no
        # iterations, just as a fixed `dof` is recorded on `dof_`
        for initial_dof, expected in ((None, 1.0), (2, 2.0)):
            with self.subTest(initial_dof=initial_dof):
                embedding = TSNE_BH(
                    dof="auto",
                    initial_dof=initial_dof,
                    early_exaggeration_iter=10,
                    n_iter=0,
                ).fit(self.x)
                self.assertEqual(embedding.dof_, expected)

    def test_initial_dof_on_optimize_restarts_learning(self):
        embedding = TSNE_BH(
            dof="auto", early_exaggeration_iter=0, n_iter=20,
        ).fit(self.x)
        self.assertNotEqual(embedding.dof_, 3.0)

        history = []
        embedding.optimize(
            n_iter=2,
            initial_dof=3,
            inplace=True,
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        self.assertEqual(history[0].dof, 3.0)

        # A restart with no iterations still moves `dof_` to `initial_dof`
        restarted = embedding.optimize(n_iter=0, initial_dof=4)
        self.assertEqual(restarted.dof_, 4.0)

    def test_learning_under_exaggeration_warns_but_learns(self):
        embedding = TSNE_BH(
            dof="auto", early_exaggeration_iter=0, n_iter=5,
        ).fit(self.x)
        for exaggeration in (4, 0.5):
            with self.subTest(exaggeration=exaggeration):
                before = embedding.dof_
                with self.assertWarnsRegex(UserWarning, "exaggeration"):
                    embedding.optimize(5, exaggeration=exaggeration, inplace=True)
                self.assertNotEqual(embedding.dof_, before)

        # The normal phase of `fit` warns too when it is exaggerated
        with self.assertWarnsRegex(UserWarning, "exaggeration"):
            TSNE_BH(
                dof="auto", early_exaggeration_iter=5, n_iter=5, exaggeration=4,
            ).fit(self.x)

    def test_fixed_dof_under_exaggeration_does_not_warn(self):
        embedding = TSNE_BH(dof=2, early_exaggeration_iter=0, n_iter=5).fit(self.x)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            embedding.optimize(5, exaggeration=4, inplace=True)

    def test_rejected_initial_dof_leaves_affinities_intact(self):
        # The exaggerated optimizer scales P in place, so validation must run
        # before that or a rejected call would corrupt the affinities
        embedding = TSNE_BH(dof="auto").prepare_initial(self.x)
        P_before = embedding.affinities.P.copy()
        with self.assertRaisesRegex(ValueError, "initial_dof"):
            embedding.optimize(5, exaggeration=12, initial_dof=-1)
        np.testing.assert_array_equal(embedding.affinities.P.data, P_before.data)

    def test_dof_lr_is_n_independent(self):
        # The same dof_lr should give comparable learned dof regardless of the
        # subsample size (the optimum can differ, but learning must not blow up).
        for n in (75, 150):
            x = np.repeat(self.x, n // len(self.x) + 1, axis=0)[:n]
            emb = TSNE_BH(dof="auto", n_iter=100).fit(x)
            self.assertTrue(0.1 < emb.dof_ < 20.0)


class TestIterationStateCallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_callback_receives_iteration_state(self):
        seen = []
        TSNE_BH(
            early_exaggeration_iter=0,
            n_iter=5,
            callbacks=seen.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertEqual(len(seen), 5)
        for state in seen:
            self.assertIsInstance(state, IterationState)
            self.assertEqual(state.embedding.shape[1], 2)
            self.assertEqual(state.gradient.shape, state.embedding.shape)
            self.assertTrue(np.isfinite(state.error))

    def test_state_embedding_is_tsne_embedding_with_dof(self):
        # Each snapshot must remain a TSNEEmbedding (so consumers can call
        # `optimizer`, `affinities`, etc. on it later) and carry the dof from
        # that iteration.
        from openTSNE import TSNEEmbedding

        seen_fixed = []
        TSNE_BH(
            dof=2.5,
            early_exaggeration_iter=0,
            n_iter=3,
            callbacks=seen_fixed.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        for state in seen_fixed:
            self.assertIsInstance(state.embedding, TSNEEmbedding)
            self.assertEqual(state.embedding.dof_, 2.5)

        seen_auto = []
        TSNE_BH(
            dof="auto",
            early_exaggeration_iter=0,
            n_iter=3,
            callbacks=seen_auto.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        for state in seen_auto:
            self.assertIsInstance(state.embedding, TSNEEmbedding)
            self.assertEqual(state.embedding.dof_, state.dof)

    def test_callback_state_is_a_copy(self):
        # Callback must receive snapshots so retained state doesn't mutate.
        seen = []
        TSNE_BH(
            early_exaggeration_iter=0,
            n_iter=3,
            callbacks=seen.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        # Embeddings on consecutive iters must differ (optimizer moved).
        self.assertFalse(np.array_equal(seen[0].embedding, seen[-1].embedding))
        # And the snapshot arrays must not alias each other.
        self.assertIsNot(seen[0].embedding, seen[-1].embedding)

    def test_retained_snapshot_keeps_embedding_metadata(self):
        import pickle

        seen = []
        embedding = TSNE_BH(
            dof=2.5,
            early_exaggeration_iter=0,
            n_iter=3,
            callbacks=seen.append,
            callbacks_every_iters=1,
        ).fit(self.x)
        snapshot = seen[0].embedding

        self.assertIs(type(seen[0].gradient), np.ndarray)
        self.assertIs(snapshot.affinities, embedding.affinities)
        self.assertEqual(snapshot.gradient_descent_params["dof"], 2.5)
        # The optimizer kept running after the snapshot, so its state must
        # not be shared with the snapshot.
        self.assertIsNot(snapshot.optimizer, embedding.optimizer)
        self.assertFalse(
            np.array_equal(
                snapshot.optimizer.optimizers["embedding"].gains,
                embedding.optimizer.optimizers["embedding"].gains,
            )
        )

        loaded = pickle.loads(pickle.dumps(snapshot))
        np.testing.assert_array_equal(loaded, snapshot)
        self.assertEqual(loaded.dof_, 2.5)

        resumed = snapshot.optimize(2)
        self.assertEqual(resumed.optimization_iters_, 3)

    def test_retained_partial_snapshot_can_be_optimized(self):
        embedding = TSNE_BH(early_exaggeration_iter=0, n_iter=5).fit(self.x)
        seen = []
        partial_embedding = embedding.prepare_partial(self.x[::10])
        partial_embedding.optimize(
            2, callbacks=seen.append, callbacks_every_iters=1, inplace=True
        )
        snapshot = seen[0].embedding
        self.assertIs(snapshot.reference_embedding, embedding)
        self.assertIs(snapshot.P, partial_embedding.P)
        resumed = snapshot.optimize(1)
        self.assertTrue(np.all(np.isfinite(resumed)))

    def test_old_three_arg_callback_still_works_with_warning(self):
        calls = []

        def old_callback(iteration, error, embedding):
            calls.append((iteration, error, embedding.shape))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TSNE_BH(
                early_exaggeration_iter=0,
                n_iter=5,
                callbacks=old_callback,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertEqual(len(calls), 5)
        self.assertTrue(
            any(issubclass(w.category, FutureWarning) for w in caught),
            f"Expected FutureWarning, got: {[w.category for w in caught]}",
        )

    def test_callback_style_detection(self):
        from openTSNE.tsne import _adapt_callback

        new_style = [
            lambda state: None,
            lambda state=None: None,
            lambda state, extra=None: None,
            lambda *args: None,
        ]
        old_style = [
            lambda iteration, error, embedding: None,
            lambda iteration, error, embedding=None: None,
            lambda iteration, error=None, embedding=None: None,
            lambda iteration, error, *args: None,
        ]
        invalid = [lambda: None, lambda a, b: None, lambda a, b, c, d: None]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            for cb in new_style:
                self.assertIs(_adapt_callback(cb), cb)
        for cb in old_style:
            with self.assertWarns(FutureWarning):
                wrapped = _adapt_callback(cb)
            self.assertIsNot(wrapped, cb)
        for cb in invalid:
            with self.assertRaises(TypeError):
                _adapt_callback(cb)

    def test_optimization_about_to_start_fires_for_new_style_callback(self):
        # Class-based callback with the new IterationState signature.
        class MyCallback:
            def __init__(self):
                self.start_calls = 0
                self.iter_calls = 0

            def optimization_about_to_start(self):
                self.start_calls += 1

            def __call__(self, state):
                self.iter_calls += 1

        cb = MyCallback()
        TSNE_BH(
            early_exaggeration_iter=5,
            n_iter=5,
            callbacks=cb,
            callbacks_every_iters=1,
        ).fit(self.x)
        # Standard fit calls .optimize() twice (early exaggeration + main).
        self.assertEqual(cb.start_calls, 2)
        self.assertGreater(cb.iter_calls, 0)

    def test_optimization_about_to_start_fires_for_old_style_callback(self):
        # Class-based callback wrapped by the deprecation adapter must still
        # have its `optimization_about_to_start` hook invoked.
        class MyOldCallback:
            def __init__(self):
                self.start_calls = 0
                self.iter_calls = 0

            def optimization_about_to_start(self):
                self.start_calls += 1

            def __call__(self, iteration, error, embedding):
                self.iter_calls += 1

        cb = MyOldCallback()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            TSNE_BH(
                early_exaggeration_iter=5,
                n_iter=5,
                callbacks=cb,
                callbacks_every_iters=1,
            ).fit(self.x)
        self.assertEqual(cb.start_calls, 2)
        self.assertGreater(cb.iter_calls, 0)


class TestOptimizationItersTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = datasets.load_iris()["data"]

    def test_initialized_to_zero(self):
        from openTSNE import TSNEEmbedding
        from openTSNE import affinity, initialization

        aff = affinity.PerplexityBasedNN(self.x, perplexity=15, random_state=42)
        init = initialization.pca(self.x, random_state=42)
        embedding = TSNEEmbedding(init, aff, negative_gradient_method="bh")
        self.assertEqual(embedding.optimization_iters_, 0)

    def test_accumulates_across_optimize_calls(self):
        embedding = TSNE_BH(
            early_exaggeration_iter=0, n_iter=10,
        ).fit(self.x)
        self.assertEqual(embedding.optimization_iters_, 10)

        embedding.optimize(n_iter=15, inplace=True)
        self.assertEqual(embedding.optimization_iters_, 25)

        embedding.optimize(n_iter=7, inplace=True)
        self.assertEqual(embedding.optimization_iters_, 32)

    def test_accumulates_across_non_inplace_optimize_calls(self):
        # The default optimize() returns a fresh embedding; the counter must
        # carry over to the new instance, not reset to zero.
        embedding = TSNE_BH(
            early_exaggeration_iter=0, n_iter=10,
        ).fit(self.x)
        self.assertEqual(embedding.optimization_iters_, 10)

        embedding2 = embedding.optimize(n_iter=15)
        self.assertEqual(embedding2.optimization_iters_, 25)
        # The original is untouched.
        self.assertEqual(embedding.optimization_iters_, 10)

        embedding3 = embedding2.optimize(n_iter=7)
        self.assertEqual(embedding3.optimization_iters_, 32)

    def test_dof_carries_through_non_inplace_optimize(self):
        # The same copy path must also carry dof_, so dof="auto" resumes from
        # the prior value across non-inplace optimize() calls (mirroring the
        # inplace=True behavior covered elsewhere).
        embedding = TSNE_BH(
            dof=5.0, early_exaggeration_iter=0, n_iter=10,
        ).fit(self.x)
        self.assertEqual(embedding.dof_, 5.0)

        history = []
        embedding.optimize(
            n_iter=2,
            dof="auto",
            callbacks=history.append,
            callbacks_every_iters=1,
        )
        self.assertEqual(history[0].dof, 5.0)

    def test_interrupt_counts_only_applied_updates(self):
        embedding = TSNE_BH(early_exaggeration_iter=0, n_iter=4).fit(self.x)
        before = np.array(embedding)

        stopped = embedding.optimize(
            5, callbacks=lambda state: True, callbacks_every_iters=1
        )
        np.testing.assert_array_equal(stopped, before)
        self.assertEqual(stopped.optimization_iters_, 4)

        seen = []

        def stop_at_third(state):
            seen.append(state.embedding.optimization_iters_)
            return state.iteration == 3

        stopped = embedding.optimize(
            5, callbacks=stop_at_third, callbacks_every_iters=1
        )
        self.assertEqual(stopped.optimization_iters_, 6)
        stopped.optimize(1, inplace=True, callbacks=stop_at_third,
                         callbacks_every_iters=1)
        # The interrupted iteration is evaluated again, under the same index
        self.assertEqual(seen, [5, 6, 7, 7])

    def test_callback_sees_monotonic_global_counter(self):
        # The snapshot's optimization_iters_ must increase monotonically across
        # consecutive optimize() calls so callbacks can plot a continuous
        # trajectory.
        seen = []

        def cb(state):
            seen.append(state.embedding.optimization_iters_)

        embedding = TSNE_BH(
            early_exaggeration_iter=0,
            n_iter=5,
            callbacks=cb,
            callbacks_every_iters=1,
        ).fit(self.x)
        self.assertEqual(seen, [1, 2, 3, 4, 5])

        embedding.optimize(
            n_iter=4,
            inplace=True,
            callbacks=cb,
            callbacks_every_iters=1,
        )
        self.assertEqual(seen, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_fit_increments_for_both_phases(self):
        # TSNE.fit calls optimize twice (early-exag + main); both must count.
        embedding = TSNE_BH(
            early_exaggeration_iter=7, n_iter=11,
        ).fit(self.x)
        self.assertEqual(embedding.optimization_iters_, 18)


class TestTransformWithLearnedDof(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = datasets.load_iris()["data"]
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(x))
        cls.x_train = x[idx[:120]]
        cls.x_new = x[idx[120:]]

    def test_transform_pins_learned_dof(self):
        # After fitting a reference embedding with `dof="auto"`, transforming
        # new points must keep dof fixed at the learned value — the new
        # points should live in the same kernel as the reference, not relearn
        # dof from scratch.
        reference = TSNE_BH(
            dof="auto", early_exaggeration_iter=5, n_iter=20,
        ).fit(self.x_train)
        learned_dof = reference.dof_
        self.assertIsInstance(learned_dof, float)
        self.assertNotEqual(learned_dof, 1.0)

        history = []

        def cb(state):
            history.append(state.dof)

        partial = reference.prepare_partial(self.x_new)
        # The partial embedding must already reflect the parent's learned dof.
        self.assertEqual(partial.dof_, learned_dof)
        # And the gradient_descent_params handed to the optimizer must be
        # `dof=<float>` rather than `"auto"`, so dof stays fixed.
        self.assertEqual(partial.gradient_descent_params["dof"], learned_dof)

        partial.optimize(
            n_iter=5,
            inplace=True,
            callbacks=cb,
            callbacks_every_iters=1,
        )
        # All callback snapshots must show the pinned dof.
        self.assertTrue(history)
        for d in history:
            self.assertEqual(d, learned_dof)

    def test_transform_smoke_with_dof_auto(self):
        # Higher-level smoke test: full transform() call on a `dof="auto"`
        # fit must run end-to-end and return an embedding whose dof_ matches
        # the reference.
        reference = TSNE_BH(
            dof="auto", early_exaggeration_iter=5, n_iter=20,
        ).fit(self.x_train)

        new_embedding = reference.transform(self.x_new)
        self.assertEqual(new_embedding.shape, (self.x_new.shape[0], 2))
        self.assertEqual(new_embedding.dof_, reference.dof_)

    def test_transform_uses_fixed_parent_dof(self):
        # Symmetric case: parent fitted with a fixed non-default dof. The
        # partial embedding (and the transformed result) must use the same
        # value — the new points should live in the same kernel.
        reference = TSNE_BH(
            dof=5.0, early_exaggeration_iter=5, n_iter=20,
        ).fit(self.x_train)
        self.assertEqual(reference.dof_, 5.0)

        partial = reference.prepare_partial(self.x_new)
        self.assertEqual(partial.dof_, 5.0)
        self.assertEqual(partial.gradient_descent_params["dof"], 5.0)

        history = []
        partial.optimize(
            n_iter=5,
            inplace=True,
            callbacks=lambda s: history.append(s.dof),
            callbacks_every_iters=1,
        )
        self.assertTrue(history)
        for d in history:
            self.assertEqual(d, 5.0)

        new_embedding = reference.transform(self.x_new)
        self.assertEqual(new_embedding.dof_, 5.0)

    def test_unoptimized_auto_parent_passes_its_starting_dof(self):
        # A parent constructed with `dof="auto"` but never optimized already
        # holds the value learning would start from in `dof_`, i.e.
        # `initial_dof` or 1, so new points use that.
        from openTSNE import TSNEEmbedding
        from openTSNE import affinity, initialization

        aff = affinity.PerplexityBasedNN(
            self.x_train, perplexity=15, random_state=42
        )
        init = initialization.pca(self.x_train, random_state=42)
        for initial_dof, expected in ((None, 1.0), (3.5, 3.5)):
            with self.subTest(initial_dof=initial_dof):
                reference = TSNEEmbedding(
                    init, aff, negative_gradient_method="bh",
                    dof="auto", initial_dof=initial_dof,
                )
                partial = reference.prepare_partial(self.x_new)
                self.assertEqual(partial.gradient_descent_params["dof"], expected)
                self.assertEqual(partial.dof_, expected)


class TestPartialDof(unittest.TestCase):
    """New points always use the dof the reference embedding was optimized
    with; dof is never learned for them."""

    @classmethod
    def setUpClass(cls):
        x = datasets.load_iris()["data"]
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(x))
        cls.x_train = x[idx[:120]]
        cls.x_new = x[idx[120:]]

        # A reference embedding fitted with `dof="auto"`. Every test that
        # needs a learned dof reuses this so we only pay the optimization
        # cost once.
        cls.auto_reference = TSNE_BH(
            dof="auto", early_exaggeration_iter=5, n_iter=20,
        ).fit(cls.x_train)
        cls.learned_dof = cls.auto_reference.dof_
        assert isinstance(cls.learned_dof, float)
        assert cls.learned_dof != 1.0

        cls.fixed_reference = TSNE_BH(
            dof=5.0, early_exaggeration_iter=5, n_iter=20,
        ).fit(cls.x_train)

    def _dof_history(self, partial):
        history = []
        partial.optimize(
            n_iter=5,
            inplace=True,
            callbacks=lambda s: history.append(s.dof),
            callbacks_every_iters=1,
        )
        return history

    def test_uses_learned_dof_of_auto_parent(self):
        partial = self.auto_reference.prepare_partial(self.x_new)
        self.assertEqual(partial.dof_, self.learned_dof)
        self.assertEqual(
            partial.gradient_descent_params["dof"], self.learned_dof
        )
        history = self._dof_history(partial)
        self.assertTrue(history)
        for d in history:
            self.assertEqual(d, self.learned_dof)

    def test_uses_dof_of_fixed_parent(self):
        partial = self.fixed_reference.prepare_partial(self.x_new)
        self.assertEqual(partial.dof_, 5.0)
        self.assertEqual(partial.gradient_descent_params["dof"], 5.0)
        for d in self._dof_history(partial):
            self.assertEqual(d, 5.0)

    def test_learning_dof_for_new_points_raises(self):
        # Against a fixed reference the objective has no interior optimum in
        # dof, so the optimizer rejects learning it
        for name, TSNE_ in (("bh", TSNE_BH), ("fft", TSNE_FFT)):
            with self.subTest(method=name):
                reference = TSNE_(
                    dof="auto", early_exaggeration_iter=5, n_iter=20,
                ).fit(self.x_train)
                partial = reference.prepare_partial(self.x_new)
                P_before = partial.P.copy()
                with self.assertRaisesRegex(ValueError, "new points"):
                    partial.optimize(5, dof="auto")
                np.testing.assert_array_equal(partial.P.data, P_before.data)

    def test_objectives_reject_dof_gradient_against_reference(self):
        partial = self.auto_reference.prepare_partial(self.x_new)
        self.auto_reference.prepare_interpolation_grid()
        common = dict(
            embedding=np.asarray(partial),
            P=partial.P,
            dof=1.5,
            reference_embedding=self.auto_reference,
            compute_dof_grad=True,
        )
        with self.assertRaisesRegex(ValueError, "reference embedding"):
            kl_divergence_bh(bh_params={"theta": 0.5}, **common)
        with self.assertRaisesRegex(ValueError, "reference embedding"):
            kl_divergence_fft(
                fft_params={
                    "n_interpolation_points": 3,
                    "min_num_intervals": 50,
                    "ints_in_interval": 1,
                },
                **common,
            )


if __name__ == "__main__":
    unittest.main()

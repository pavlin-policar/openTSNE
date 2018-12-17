"""Bencharking module

Run something like this:

    python benchmark.py openTSNEapprox run_multiple --n-samples 1000 --n 10     > logs/opentsne_approx_fft_1000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 5000 --n 10     > logs/opentsne_approx_fft_5000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 10000 --n 10    > logs/opentsne_approx_fft_10000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 20000 --n 10    > logs/opentsne_approx_fft_20000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 40000 --n 10    > logs/opentsne_approx_fft_40000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 100000 --n 10   > logs/opentsne_approx_fft_100000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 200000 --n 10   > logs/opentsne_approx_fft_200000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 500000 --n 10   > logs/opentsne_approx_fft_500000.log 2>&1
    python benchmark.py openTSNEapprox run_multiple --n-samples 1000000 --n 10  > logs/opentsne_approx_fft_1000000.log 2>&1

    python benchmark.py openTSNEexact run_multiple --n-samples 1000 --n 10     > logs/opentsne_exact_bh_1000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 5000 --n 10     > logs/opentsne_exact_bh_5000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 10000 --n 10    > logs/opentsne_exact_bh_10000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 20000 --n 10    > logs/opentsne_exact_bh_20000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 40000 --n 10    > logs/opentsne_exact_bh_40000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 100000 --n 10   > logs/opentsne_exact_bh_100000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 200000 --n 10   > logs/opentsne_exact_bh_200000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 500000 --n 10   > logs/opentsne_exact_bh_500000.log 2>&1
    python benchmark.py openTSNEexact run_multiple --n-samples 1000000 --n 10  > logs/opentsne_exact_bh_1000000.log 2>&1
    
    python benchmark.py MulticoreTSNE run_multiple --n-samples 1000 --n 10      > logs/multicore_exact_bh_1000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 5000 --n 10      > logs/multicore_exact_bh_5000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 10000 --n 10     > logs/multicore_exact_bh_10000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 20000 --n 10     > logs/multicore_exact_bh_20000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 40000 --n 10     > logs/multicore_exact_bh_40000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 100000 --n 10    > logs/multicore_exact_bh_100000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 200000 --n 10    > logs/multicore_exact_bh_200000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 500000 --n 10    > logs/multicore_exact_bh_500000.log 2>&1
    python benchmark.py MulticoreTSNE run_multiple --n-samples 1000000 --n 10   > logs/multicore_exact_bh_1000000.log 2>&1

    python benchmark.py FItSNE run_multiple --n-samples 1000 --n 10     > logs/fitsne_approx_fft_1000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 5000 --n 10     > logs/fitsne_approx_fft_5000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 10000 --n 10    > logs/fitsne_approx_fft_10000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 20000 --n 10    > logs/fitsne_approx_fft_20000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 40000 --n 10    > logs/fitsne_approx_fft_40000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 100000 --n 10   > logs/fitsne_approx_fft_100000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 200000 --n 10   > logs/fitsne_approx_fft_200000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 500000 --n 10   > logs/fitsne_approx_fft_500000.log 2>&1
    python benchmark.py FItSNE run_multiple --n-samples 1000000 --n 10  > logs/fitsne_approx_fft_1000000.log 2>&1

    python benchmark.py sklearn run_multiple --n-samples 1000 --n 10     > logs/sklearn_exact_bh_1000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 5000 --n 10     > logs/sklearn_exact_bh_5000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 10000 --n 10    > logs/sklearn_exact_bh_10000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 20000 --n 10    > logs/sklearn_exact_bh_20000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 40000 --n 10    > logs/sklearn_exact_bh_40000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 100000 --n 10   > logs/sklearn_exact_bh_100000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 200000 --n 10   > logs/sklearn_exact_bh_200000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 500000 --n 10   > logs/sklearn_exact_bh_500000.log 2>&1
    python benchmark.py sklearn run_multiple --n-samples 1000000 --n 10  > logs/sklearn_exact_bh_1000000.log 2>&1

"""
import gzip
import pickle
import sys
import time
from os import path

import fire
import numpy as np
from MulticoreTSNE import MulticoreTSNE as MulticoreTSNE_
from fitsne import FItSNE as FItSNE_
from sklearn.manifold import TSNE as SKLTSNE

import openTSNE


class TSNEBenchmark:
    perplexity = 30
    learning_rate = 100
    n_jobs = 1

    def run(self, n_samples=1000):
        raise NotImplemented()

    def run_multiple(self, n=5, n_samples=1000):
        for _ in range(n):
            self.run(n_samples=n_samples)

    def load_data(self, n_samples=None):
        with gzip.open(path.join("data", "10x_mouse_zheng.pkl.gz"), "rb") as f:
            data = pickle.load(f)

        x, y = data["pca_50"], data["CellType1"]

        if n_samples is not None:
            indices = np.random.choice(list(range(x.shape[0])), n_samples, replace=False)
            x, y = x[indices], y[indices]

        return x, y


class openTSNEapprox(TSNEBenchmark):
    def run(self, n_samples=1000):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x, perplexity=self.perplexity, method="approx", n_jobs=self.n_jobs,
        )
        print("openTSNE: NN search", time.time() - start_aff)

        init = openTSNE.initialization.random(x.shape[0], n_components=2)

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, learning_rate=self.learning_rate, n_jobs=self.n_jobs,
            negative_gradient_method="fft", theta=0.5,
            min_num_intervals=10, ints_in_interval=1,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start)


class openTSNEexact(TSNEBenchmark):
    def run(self, n_samples=1000):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x, perplexity=self.perplexity, method="exact", n_jobs=self.n_jobs,
        )
        print("openTSNE: NN search", time.time() - start_aff)

        init = openTSNE.initialization.random(x.shape[0], n_components=2)

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init, affinity, learning_rate=self.learning_rate, n_jobs=self.n_jobs,
            negative_gradient_method="bh", theta=0.5,
            min_num_intervals=10, ints_in_interval=1,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start)


class MulticoreTSNE(TSNEBenchmark):
    def run(self, n_samples=1000):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80), sys.stdout.flush()
        start = time.time()
        tsne = MulticoreTSNE_(
            early_exaggeration=12, learning_rate=self.learning_rate,
            perplexity=self.perplexity, n_jobs=self.n_jobs, angle=0.5,
            verbose=True,
        )
        tsne.fit_transform(x)
        print("Multicore t-SNE:", time.time() - start), sys.stdout.flush()


class FItSNE(TSNEBenchmark):
    def run(self, n_samples=1000):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80), sys.stdout.flush()
        start = time.time()
        FItSNE_(
            x, 2, perplexity=self.perplexity, stop_lying_iter=250, ann_not_vptree=True,
            early_exag_coeff=12, nthreads=1, theta=0.5,
        )
        print("FIt-SNE:", time.time() - start), sys.stdout.flush()
        
        
class sklearn(TSNEBenchmark):
    def run(self, n_samples=1000):
        x, y = self.load_data(n_samples=n_samples)
        print("-" * 80)
        start = time.time()
        SKLTSNE(
            early_exaggeration=12, learning_rate=self.learning_rate, angle=0.5,
            perplexity=self.perplexity, init="random", verbose=True,
        ).fit_transform(x)
        print("scikit-learn t-SNE:", time.time() - start)


if __name__ == "__main__":
    fire.Fire()

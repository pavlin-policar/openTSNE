"""Bencharking module

Run something like this:

Choose one of the available methods:
--> openTSNEapprox | openTSNEexact | MulticoreTSNE | FItSNE | sklearn

METHOD="openTSNEapprox";
SAMPLE_SIZES=(1000 5000 10000 100000 250000 500000 750000 1000000);
REPETITIONS=3;

for size in ${SAMPLE_SIZES[@]}; do
    cmd="python benchmark.py $METHOD run_multiple --n-samples $size --n $REPETITIONS 2>&1 | tee -a logs/${METHOD}_${size}.log";
    echo "$cmd";
    eval "$cmd";
done;

"""
import gzip
import pickle
import time
from os import path

import fire
import numpy as np
from MulticoreTSNE import MulticoreTSNE as MulticoreTSNE_
from sklearn.manifold import TSNE as SKLTSNE
from sklearn.utils import check_random_state

import openTSNE
import openTSNE.callbacks


class TSNEBenchmark:
    perplexity = 30
    learning_rate = 100
    n_jobs = 1

    def run(self, n_samples=1000, random_state=None):
        raise NotImplementedError()

    def run_multiple(self, n=5, n_samples=1000):
        for idx in range(n):
            self.run(n_samples=n_samples, random_state=idx)

    def load_data(self, n_samples=None):
        with gzip.open(path.join("data", "10x_mouse_zheng.pkl.gz"), "rb") as f:
            data = pickle.load(f)

        x, y = data["pca_50"], data["CellType1"]

        if n_samples is not None:
            indices = np.random.choice(
                list(range(x.shape[0])), n_samples, replace=False
            )
            x, y = x[indices], y[indices]

        return x, y


class openTSNEapprox(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="approx",
            n_jobs=self.n_jobs,
            random_state=random_state,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            negative_gradient_method="fft",
            callbacks=[openTSNE.callbacks.ErrorLogger()],
            random_state=random_state,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start, flush=True)


class openTSNEexact(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="exact",
            n_jobs=self.n_jobs,
            random_state=random_state,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            negative_gradient_method="bh",
            theta=0.5,
            min_num_intervals=10,
            ints_in_interval=1,
            random_state=random_state,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", time.time() - start, flush=True)


class MulticoreTSNE(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80, flush=True)
        start = time.time()
        tsne = MulticoreTSNE_(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            perplexity=self.perplexity,
            n_jobs=self.n_jobs,
            angle=0.5,
            verbose=True,
            random_state=random_state,
        )
        tsne.fit_transform(x)
        print("Multicore t-SNE:", time.time() - start, flush=True)


class FItSNE(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=-1):
        import sys

        sys.path.append("FIt-SNE")
        from fast_tsne import fast_tsne

        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        if random_state == -1:
            init = openTSNE.initialization.random(x, n_components=2)
        else:
            init = openTSNE.initialization.random(
                x, n_components=2, random_state=random_state
            )

        start = time.time()
        fast_tsne(
            x,
            map_dims=2,
            initialization=init,
            perplexity=self.perplexity,
            stop_early_exag_iter=250,
            early_exag_coeff=12,
            nthreads=self.n_jobs,
            seed=random_state,
        )
        print("FIt-SNE:", time.time() - start, flush=True)


class sklearn(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start = time.time()
        SKLTSNE(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            angle=0.5,
            perplexity=self.perplexity,
            init=init,
            verbose=True,
            random_state=random_state
        ).fit_transform(x)
        print("scikit-learn t-SNE:", time.time() - start, flush=True)


class UMAP(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        import umap

        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        start = time.time()
        umap.UMAP(random_state=random_state).fit_transform(x)
        print("UMAP:", time.time() - start, flush=True)


if __name__ == "__main__":
    fire.Fire()

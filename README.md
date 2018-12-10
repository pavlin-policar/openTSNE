# fastTSNE

[![Build Status](https://travis-ci.com/pavlin-policar/fastTSNE.svg?branch=master)](https://travis-ci.com/pavlin-policar/fastTSNE)
[![Build status](https://ci.appveyor.com/api/projects/status/2s1cbbsk8dltte3y?svg=true)](https://ci.appveyor.com/project/pavlin-policar/fasttsne)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ef67c21a74924b548acae5a514bc443d)](https://app.codacy.com/app/pavlin-policar/fastTSNE?utm_source=github.com&utm_medium=referral&utm_content=pavlin-policar/fastTSNE&utm_campaign=Badge_Grade_Dashboard)

A visualization of 160,796 single cell transcriptomes from the mouse nervous system [Zeisel 2018] computed in under 2 minutes using FFT accelerated interpolation and approximate nearest neighbors. See basic usage notebook for more details.

![Zeisel 2018 mouse nervous system t-SNE embedding](docs/source/images/zeisel_2018.png)

The goal of this project is to have fast implementations of t-SNE in one place, with a flexible API and without any external dependencies. This makes it very easy to experiment with various aspects of t-SNE and makes the package very easy to distribute.

This package provides two fast implementations of t-SNE:
1. Barnes-hut t-SNE [2] is appropriate for small data sets and has asymptotic complexity O(n log n).
2. FFT Accelerated t-SNE [3] is appropriate for larger data sets (>10,000 samples). It has asymptotic complexity O(n).

Barnes-Hut tends to be slightly faster on smaller data sets (typically by a minute or two) while FIt-SNE should always be used for larger data sets (>10,000 samples). In most cases, using the FIt-SNE implementation is a safe default.

To better understand the speed trade-offs, it is useful to know how t-SNE works. t-SNE runs in two main phases. In the first phase we find the K nearest neighbors for each sample. We offer exact nearest neighbor search using scikit-learn's nearest neighbors KDTrees and approximate nearest neighbor search using a Python/Numba implementation of nearest neighbor descent. Exact search tends to be faster for smaller data sets and approximate search is faster for larger data sets.
The second phase runs the optimization phase (which can, again, be run in several phases). In every iteration we must evaluate the negative gradient, which involves computing all pairwise interactions. This can be accelerated using Barnes-Hut space partitioning trees (scaling with O(n log n)) or FFT accelerated interpolation (scaling with O(n)) for larger data sets. For more details, see the corresponding papers.

Documentation is avaialble on [Read the Docs](http://fasttsne.readthedocs.io).

## Benchmarks
The numbers are not exact. The benchmarks were run on an Intel i7-7700HQ CPU @ 2.80GHz (up to 3.80GHz) processor.

FFT benchmarks are run using approximate nearest neigbhor search. Exact search is used for Barnes-Hut.

The typical benchmark to use is the MNIST data set containing 70,000 28x28 images (784 pixels).

| MNIST | Exact NN | Approximate NN | BH gradient | FFT gradient |
|:---|---:|---:|---:|---:|
| 4 cores | 2086s | 22s | 243s | 67s |

## Installation

fastTSNE can be installed using `conda` from conda-forge with

```
conda install --channel conda-forge fasttsne
```

fastTSNE can also be installed using pip. The only prerequisite is `numpy`.

Once numpy is installed, simply run
```
pip install fasttsne
```
and you're good to go.

### FFTW
By default, fastTSNE uses numpy's implementation of the Fast Fourier Transform because of it's wide availability. If you would like to squeeze out maximum performance, you can install the highly optimized FFTW C library, available through conda. fastTSNE will automatically detect FFTW and will use that. The speed ups here are generally not large, but can save seconds to minutes when running t-SNE on larger data sets.
 
## Usage
We provide two modes of usage. One is somewhat familliar to scikit-learn's `TSNE.fit`.

We also provide an advanced interface for finer control of the optimization, allowing us to interactively tune the embedding and make use of various tricks to improve the embedding quality.

### Basic usage

We provide a basic interface somewhat similar to the one provided by scikit-learn.

```python
from fastTSNE import TSNE
from sklearn import datasets

iris = datasets.load_iris()
x, y = iris['data'], iris['target']

tsne = TSNE(
    n_components=2, perplexity=30, learning_rate=200,
    n_jobs=4, angle=0.5, initialization='pca', metric='euclidean',
    early_exaggeration_iter=250, early_exaggeration=12, n_iter=750,
    neighbors='exact', negative_gradient_method='bh',
)

embedding = tsne.fit(x)
```

There are two parameters which you will want to watch out for:
1. `neighbors` controls nearest neighbor search. If our data set is small, `exact` is the better choice. `exact` uses scikit-learn's KD trees. For larger data, approximate search can be orders of magnitude faster. This is selected with `approx`. Nearest neighbor search is performed only once at the beginning of the optmization, but can dominate runtime on large data sets, therefore this must be properly chosen.
2. `negative_gradient_method` controls which approximation technique to use to approximate pairwise interactions. These are computed at each step of the optimization. Van Der Maaten [2] proposed using the Barnes-Hut tree approximation and this has be the de-facto standard in most t-SNE implementations. This can be selected by passing `bh`. Asymptotically, this scales as O(n log n) in the number of points works well for up to 10,000 samples. More recently, Linderman et al. [3] developed another approximation using interpolation which scales linearly in the number of points O(n). This can be selected by passing `fft`. There is a bit of overhead to this method, making it slightly slower than Barnes-Hut for small numbers of points, but is very fast for larger data sets, while Barnes-Hut becomes completely unusable. For smaller data sets the difference is typically in the order of seconds, at most minutes, so a safe default is using the FFT approximation.

Our `tsne` object acts as a fitter instance, and returns a `TSNEEmbedding` instance. This acts as a regular numpy array, and can be used as such, but can be further optimized if we see fit or can be used for adding new points to the embedding.

We don't log any progress by default, but provide callbacks that can be run at any interval of the optimization process. A simple logger is provided as an example.

```python
from fastTSNE.callbacks import ErrorLogger

tsne = TSNE(callbacks=ErrorLogger(), callbacks_every_iters=50)
```

A callback can be any callable object that accepts the following arguments.
```python
def callback(iteration, error, embedding):
    ...
```

Callbacks are used to control the optimization i.e. every callback must return a boolean value indicating whether or not to stop the optimization. If we want to stop the optimization via callback we simply return `True`.

Additionally, a list of callbacks can also be passed, in which case all the callbacks must agree to continue the optimization, otherwise the process is terminated and the current embedding is returned.

### Advanced usage

Recently, Kobak and Berens [4] demonstrate several tricks we can use to obtain better t-SNE embeddings. The main critique of t-SNE is that global structure is mainly thrown away. This is typically the main selling point for UMAP over t-SNE. In the preprint, several techniques are presented that enable t-SNE to capture more global structure. All of these tricks can easily be implemented using fastTSNE and are shown in the notebook examples.

To introduce the API, we will implement the standard t-SNE algorithm, the one implemented by `TSNE.fit`. 

```python
from fastTSNE import initialization, affinity
from fastTSNE.tsne import TSNEEmbedding

init = initialization.pca(x)
affinities = affinity.PerplexityBasedNN(x, perplexity=30, method='approx', n_jobs=8)
embedding = TSNEEmbedding(
    init, affinities, negative_gradient_method='fft',
    learning_rate=200, n_jobs=8, callbacks=ErrorLogger(),
)
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
embedding.optimize(n_iter=750, momentum=0.8, inplace=True)
```


## References

1. Maaten, Laurens van der, and Geoffrey Hinton. ["Visualizing data using t-SNE."](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) Journal of machine learning research 9.Nov (2008): 2579-2605.

2. Van Der Maaten, Laurens. ["Accelerating t-SNE using tree-based algorithms."](http://www.jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf) The Journal of Machine Learning Research 15.1 (2014): 3221-3245.

3. Linderman, George C., et al. ["Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding."](https://arxiv.org/pdf/1712.09005.pdf) arXiv preprint arXiv:1712.09005 (2017).

4. Kobak, Dmitry, and Philipp Berens. ["The art of using t-SNE for single-cell transcriptomics."](https://www.biorxiv.org/content/early/2018/10/25/453449) bioRxiv (2018): 453449.

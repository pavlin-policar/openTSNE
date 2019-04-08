# openTSNE

[![Build Status](https://travis-ci.com/pavlin-policar/openTSNE.svg?branch=master)](https://travis-ci.com/pavlin-policar/openTSNE)
[![Build status](https://ci.appveyor.com/api/projects/status/6i5vv7b7ot6iws90?svg=true)](https://ci.appveyor.com/project/pavlin-policar/opentsne/branch/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ef67c21a74924b548acae5a514bc443d)](https://app.codacy.com/app/pavlin-policar/openTSNE?utm_source=github.com&utm_medium=referral&utm_content=pavlin-policar/openTSNE&utm_campaign=Badge_Grade_Dashboard)

A visualization of 44,808 single cell transcriptomes from the mouse retina [5] embedded using the multiscale kernel trick for preserving global structure.

![Macosko 2015 mouse retina t-SNE embedding](docs/source/images/macosko_2015.png)

The goal of this project is

1. **Extensibility**. We provide efficient defaults for the typical use case i.e. visualizing high dimensional data. We also make it very simple to use various tricks that have been introduced to improve the quality of t-SNE embeddings. The library is designed to it's easy to implement and use your own components and encourages experimentation.

2. **Speed**. We provide two fast, parallel implementations of t-SNE, which are comparable to their C++ counterparts in speed. Python does incur some overhead, so if speed is your only requirement, consider using [FIt-SNE](https://github.com/KlugerLab/FIt-SNE). The differences are often minute and become even less apparent when utilizing multiple cores. 

3. **Interactivity**. This library was built for Orange, an interactive machine learning toolkit. As such, we provide a powerful API which can control all aspects of the t-SNE algorithm and makes it suitable for interactive environments.

4. **Ease of distribution**. FIt-SNE, the reference C++ implementation for the interpolation based variant of t-SNE, is not easy to install or distribute. It requires one to preinstall C libraries and requires manual compilation. This package is installable either through `pip` or `conda` with a single command, making it very easy to include in other packages.

Detailed documentation on t-SNE is available on [Read the Docs](http://opentsne.readthedocs.io).

## Installation

### Conda

openTSNE can be easily installed from ``conda-forge`` with

```
conda install --channel conda-forge opentsne
```

[Conda package](https://anaconda.org/conda-forge/opentsne)

### PyPi

openTSNE is also available through ``pip`` and can be installed with

```
pip install opentsne
```

[PyPi package](https://pypi.org/project/openTSNE)

Note that openTSNE requires a C/C++ compiler. ``numpy`` must also be installed.

In order for openTSNE to utilize multiple threads, the C/C++ compiler must also implement ``OpenMP``. In practice, almost all compilers implement this with the exception of older version of ``clang`` on OSX systems.

To squeeze the most out of openTSNE, you may also consider installing FFTW3 prior to installation. FFTW3 implements the Fast Fourier Transform, which is heavily used in openTSNE. If FFTW3 is not available, openTSNE will use numpy's implementation of the FFT, which is slightly slower than FFTW. The difference is only noticeable with large data sets containing millions of data points.

 
## Usage

We provide two modes of usage. One is somewhat familliar to scikit-learn's `TSNE.fit`.

We also provide an advanced interface for finer control of the optimization, allowing us to interactively tune the embedding and make use of various tricks to improve the embedding quality.

### Basic usage

We provide a basic interface somewhat similar to the one provided by scikit-learn.

```python
from openTSNE import TSNE
from sklearn import datasets

iris = datasets.load_iris()
x, y = iris["data"], iris["target"]

tsne = TSNE(
    n_components=2, perplexity=30, learning_rate=200,
    n_jobs=4, angle=0.5, initialization="pca", metric="euclidean",
    early_exaggeration_iter=250, early_exaggeration=12, n_iter=750,
    neighbors="exact", negative_gradient_method="bh",
)

embedding = tsne.fit(x)
```

There are two parameters which you will want to watch out for:
1. `neighbors` controls nearest neighbor search. If our data set is small, `exact` is the better choice. `exact` uses scikit-learn's KD trees. For larger data, approximate search can be orders of magnitude faster. This is selected with `approx`. Nearest neighbor search is performed only once at the beginning of the optmization, but can dominate runtime on large data sets, therefore this must be properly chosen.
2. `negative_gradient_method` controls which approximation technique to use to approximate pairwise interactions. These are computed at each step of the optimization. Van Der Maaten [2] proposed using the Barnes-Hut tree approximation and this has be the de-facto standard in most t-SNE implementations. This can be selected by passing `bh`. Asymptotically, this scales as O(n log n) in the number of points works well for up to 10,000 samples. More recently, Linderman et al. [3] developed another approximation using interpolation which scales linearly in the number of points O(n). This can be selected by passing `fft`. There is a bit of overhead to this method, making it slightly slower than Barnes-Hut for small numbers of points, but is very fast for larger data sets, while Barnes-Hut becomes completely unusable. For smaller data sets the difference is typically in the order of seconds, at most minutes, so a safe default is using the FFT approximation.

Our `tsne` object acts as a fitter instance, and returns a `TSNEEmbedding` instance. This acts as a regular numpy array, and can be used as such, but can be further optimized if we see fit or can be used for adding new points to the embedding.

We don't log any progress by default, but provide callbacks that can be run at any interval of the optimization process. A simple logger is provided as an example.

```python
from openTSNE.callbacks import ErrorLogger

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

Recently, Kobak and Berens [4] demonstrate several tricks we can use to obtain better t-SNE embeddings. The main critique of t-SNE is that global structure is mainly thrown away. This is typically the main selling point for UMAP over t-SNE. In the preprint, several techniques are presented that enable t-SNE to capture more global structure. All of these tricks can easily be implemented using openTSNE and are shown in the notebook examples.

To introduce the API, we will implement the standard t-SNE algorithm, the one implemented by `TSNE.fit`. 

```python
from openTSNE import initialization, affinity
from openTSNE.tsne import TSNEEmbedding

init = initialization.pca(x)
affinities = affinity.PerplexityBasedNN(x, perplexity=30, method="approx", n_jobs=8)
embedding = TSNEEmbedding(
    init, affinities, negative_gradient_method="fft",
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

5. Macosko, Evan Z., et al. ["Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets."](https://www.sciencedirect.com/science/article/pii/S0092867415005498) Cell 161.5 (2015): 1202-1214.

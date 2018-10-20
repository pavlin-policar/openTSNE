# fastTSNE

[![Build Status](https://travis-ci.com/pavlin-policar/fastTSNE.svg?branch=master)](https://travis-ci.com/pavlin-policar/fastTSNE)
[![Build status](https://ci.appveyor.com/api/projects/status/2s1cbbsk8dltte3y?svg=true)](https://ci.appveyor.com/project/pavlin-policar/fasttsne)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ef67c21a74924b548acae5a514bc443d)](https://app.codacy.com/app/pavlin-policar/fastTSNE?utm_source=github.com&utm_medium=referral&utm_content=pavlin-policar/fastTSNE&utm_campaign=Badge_Grade_Dashboard)

A visualization of 160,796 single cell transcriptomes from the mouse nervous system [Zeisel 2018] computed in under 2 minutes using FFT accelerated interpolation and approximate nearest neighbors. See basic usage notebook for more details.

![Zeisel 2018 mouse nervous system tSNE embedding](images/zeisel_2018.png)

The goal of this project is to have fast implementations of tSNE in one place, without any external C/C++ dependencies. This makes the package very easy to include in other projects.

This package provides two fast implementations of tSNE:
1. Barnes-hut tsne [2] is inspired by Multicore tSNE and is appropriate for small data sets and has asymptotic complexity O(n log n).
2. Fit-SNE [3] is inspired by the C++ implementation of Fit-SNE and is appropriate for larger data sets (>10,000 samples). It has asymptotic complexity O(n). The C++ implementation depends on FFTW for fast fourier transforms which must be installed independently. This makes messy for distribution.

We include both these implementations because Barnes-Hut tSNE tends to be slightly faster for smaller data sets, while Fit-SNE is much faster for larger data sets (>10,000 samples). The difference is typically in the order of seconds, at most minutes, so a safe default is using the FFT approximation.

tSNE runs in two phases. In the first phase, K nearest neighbors must be found for each sample. We offer exact nearest neighbor search using scikit-learn's nearest neighbors KDTrees and approximate nearest neighbor search using a Python/Numba implementation of nearest neighbor descent. Again, exact search is faster for smaller data sets and approximate search is faster for larger data sets.
The second phase runs the actual optimization. In each iteration the negative gradient must be computed w.r.t. the embedding. This can be computed using Barnes-Hut space partitioning trees or FFT accelerated interpolation. For more details, see the corresponding papers.

## Benchmarks
The numbers are not exact. The benchmarks were run on an Intel i7-7700HQ CPU @ 2.80GHz (up to 3.80GHz) processor.

FFT benchmarks are run using approximate nearest neigbhor search. Exact search is used for Barnes-Hut.

The typical benchmark to use is the MNIST data set containing 70,000 28x28 images (784 pixels).

| MNIST | Exact NN | Approximate NN | BH gradient | FFT gradient |
|:---|---:|---:|---:|---:|
| 4 cores | 2086s | 22s | 243s | 67s |

## Installation

The only prerequisite is `numpy`. This is necessary so we can link against numpy header files in cython.

Once numpy is installed, simply run
```
pip install fasttsne
```
and you're good to go.
 
## Usage
We provide two modes of usage. One is somewhat familliar to scikit-learn's `TSNE.fit`.

We also provide an advanced interface for much more control of the optimization process, allowing us to interactively tune the embedding.

### Basic usage

Can be used with any numpy arrays. The interface is similar to the one provided by scikit-learn.

```python
from fastTSNE import TSNE
from sklearn import datasets

iris = datasets.load_iris()
x, y = iris['data'], iris['target']

tsne = TSNE(
	n_components=2, perplexity=30, learning_rate=100, early_exaggeration=12,
	n_jobs=4, angle=0.5, initialization='random', metric='euclidean',
	n_iter=750, early_exaggeration_iter=250, neighbors='exact',
	negative_gradient_method='bh', min_num_intervals=10,
	ints_in_inverval=2, late_exaggeration_iter=100, late_exaggeration=4,
)

embedding = tsne.fit(x)
```

There are two parameters which greatly impact the runtime:
1. `neighbors` controls nearest neighbor search. If our data are small, `exact` is the better choice. `exact` uses scikit-learn's KD trees. For larger data, approximate search can be orders of magnitude faster. This is selected with `approx`. Nearest neighbor search is performed only once at the beginning of the optmization, but can dominate runtime on large data sets, therefore this must be properly chosen.
2. `negative_gradient_method` controls which approximation technique to use to approximate gradients. Gradients are computed at each step of the optimization. Van Der Maaten [2] proposed using the Barnes-Hut tree approximation and this has be the de-facto standard in most tSNE implementations. This can be selected by passing `bh`. Asymptotically, this scales as O(n log n) in the number of points works well for up to 10,000 samples. More recently, Linderman et al. [3] developed another approximation using interpolation which scales linearly in the number of points O(n). This can be selected by passing `fft`. There is a bit of overhead to this method, making it slightly slower than Barnes-Hut for small numbers of points, but is very fast for larger data sets. The difference is typically in the order of seconds, at most minutes, so a safe default is using the FFT approximation.

tSNE optimization is typically run in two phases. The first phase is called the *early exaggeration* phase. In this phase, we exaggerate how close similar points should be to allow for better grouping and correct for bad initializations. The second phase runs tSNE optimization with no exaggeration. Theoretically, we could pick and choose these as many times and in whatever way we want. Linderman et al. [3] recently propose running another exaggerated phase after the normal phase so the clusters are more tightly packed.

Our `tsne` object acts as a fitter instance, and returns a `TSNEEmbedding` instance. This acts as a regular numpy array, and can be used as such, but can be further optimized if we see fit or can be used for adding new points to the embedding.

We don't log any progress by default, but provide callbacks that can be run at any interval of the optimization process. A simple logger is provided as an example.

```python
from fastTSNE.callbacks import ErrorLogger

tsne = TSNE(callbacks=ErrorLogger(), callbacks_every_iters=50)
```

The callback can be any callable object that accepts the following arguments.
```python
def callback(iteration, error, embedding):
    ...
```

Callbacks are used to control the optimization i.e. every callback must return a boolean value indicating whether or not to stop the optimization. If we want to stop the optmimization via callback we simply return `True`.

Additionally, a list of callbacks can also be passed, in which case all the callbacks must agree to continue the optimization, otherwise the process is terminated and the current embedding is returned.

### Advanced usage

If we want finer control of the optimization process, we can run individual optimization phases (early/late exaggeration) as desired. A typical run of tSNE in scikit-learn using this interface is implemented as follows:

```python
tsne = TSNE()
embedding = tsne.prepare_initial(x)
embedding = embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
embedding = embedding.optimize(n_iter=750, momentum=0.8)
```

Note that all the aspects of optimization can be controlled via the `.optimize` method, see the docs for an extensive list of parameters.


## Future work

- Automatically determine which nearest neighbor/gradient method to use depending on the data set size.

## References

1. Maaten, Laurens van der, and Geoffrey Hinton. ["Visualizing data using t-SNE."](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) Journal of machine learning research 9.Nov (2008): 2579-2605.

2. Van Der Maaten, Laurens. ["Accelerating t-SNE using tree-based algorithms."](http://www.jmlr.org/papers/volume15/vandermaaten14a/vandermaaten14a.pdf) The Journal of Machine Learning Research 15.1 (2014): 3221-3245.

3. Linderman, George C., et al. ["Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding."](https://arxiv.org/pdf/1712.09005.pdf) arXiv preprint arXiv:1712.09005 (2017).

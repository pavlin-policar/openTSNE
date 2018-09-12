# tSNE



[![Build Status](https://travis-ci.com/pavlin-policar/fastTSNE.svg?branch=master)](https://travis-ci.com/pavlin-policar/fastTSNE)
[![Build status](https://ci.appveyor.com/api/projects/status/2s1cbbsk8dltte3y?svg=true)](https://ci.appveyor.com/project/pavlin-policar/fasttsne)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ef67c21a74924b548acae5a514bc443d)](https://app.codacy.com/app/pavlin-policar/fastTSNE?utm_source=github.com&utm_medium=referral&utm_content=pavlin-policar/fastTSNE&utm_campaign=Badge_Grade_Dashboard)

The goal of this project is to have fast implementations of tSNE in one place, without any external C/C++ dependencies. This makes the package very easy to include in other projects.

This package provides two fast implementations of tSNE:
1. Barnes-hut tsne [2] is inspired by Multicore tSNE and is appropriate for small data sets and has asymptotic complexity O(n log n).
2. Fit-SNE [3] is inspired by the C++ implementation of Fit-SNE and is appropriate for larger data sets (>10,000 samples). It has asymptotic complexity O(n). The C++ implementation depends on FFTW for fast fourier transforms which must be installed independently. This makes messy for distribution.

We include both these implementations because Barnes-Hut tSNE tends to be much faster for smaller data sets, while Fit-SNE is much faster for larger data sets (>10,000 samples).

tSNE runs in two phases. In the first phase, K nearest neighbors must be found for each sample. We offer exact nearest neighbor search using scikit-learn's nearest neighbors KDTrees and approximate nearest neighbor search using a Python/Numba implementation of nearest neighbor descent. Again, exact search is faster for smaller data sets and approximate search is faster for larger data sets.
The second phase runs the actual optimization. In each iteration the negative gradient must be computed w.r.t. the embedding. This can be computed using Barnes-Hut space partitioning trees or FFT accelerated interpolation. For more details, see the corresponding papers.

## Benchmarks
The numbers are not exact. The benchmarks were run on an Intel i7-7700HQ CPU @ 2.80GHz (up to 3.80GHz) processor.

FFT benchmarks are run using approximate nearest neigbhor search. Exact search is used for Barnes-Hut.

The typical benchmark to use is the MNIST data set containing 70,000 28x28 images (784 pixels).

| MNIST | Exact NN | Approximate NN | BH gradient | FFT gradient |
|:---|---:|---:|---:|---:|
| 4 cores | 2086s | 22s | 243s | 67s |

 
## Usage
We provide two modes of usage. One is very familliar to anyone who has ever used scikit-learn via `TSNE.fit`.

We also provide an advanced interface for much more control of the optimization process, allowing us to interactively tune the embedding.

### Basic usage

Can be used with any numpy arrays. The interface is similar to the one provided by scikit-learn.

```python
from sklearn import datasets

iris = datasets.load_iris()
x = iris['data']
y = iris['target']

tsne = TSNE(
	n_components=2, perplexity=30, learning_rate=100, early_exaggeration=12,
	n_jobs=4, angle=0.5, initialization='pca', metric='euclidean',
	n_iter=750, early_exaggeration_iter=250, neighbors='exact',
	negative_gradient_method='bh', min_num_intervals=10,
	ints_in_inverval=2, late_exaggeration_iter=100, late_exaggeration=4,
)

embedding = tsne.fit(x)
```

There are some key differences from scikit-learn. Scikit-learn offers tSNE optimization in two phases: the early exaggerated and normal regime, and therefore the number of iterations for the normal regime is `n_iter` - `early_exaggeration_iter`. Our interface does not do this but actually runs the number of iterations in the provided regime. This implementation also offers a `late_exaggeration` regime that runs after the normal regime. This is optional and sometimes improves separation of clusters, since the attractive forces are scaled up.

Another key difference is that we return a `TSNEEmbedding` instance. This acts as a regular numpy array, and can be used as such, but can be used further on for adding new points to the embedding, as described later on. The instance also contains the KL divergence and the pBIC as attributes, as opposed to scikit-learn, which stores these attributes on the TSNE fitter itself. This allows us to reuse the fitter for multiple embeddings while still having access to the embedding error.

We don't log any progress by default, but provide callbacks that can be run at any interval of the optimization process. A simple logger is provided as an example.

```python
from fastTSNE.callbacks import ErrorLogger

tsne = TSNE(callbacks=ErrorLogger(), callbacks_every_iters=50)
```

In this instance, the callback is a callable object, but any function that accepts the following parameters is valid.
```python
def callback(iteration, error, embedding):
	...
```

Callbacks are used to control the optimization i.e. every callback must return a boolean value indicating whether or not to stop the optimization. We return `True` if we want to stop. This is convenient because if we want a logging callback, it is easy to forget the return value, and optimization proceeds as planned.

Additionally, a list of callbacks can also be passed, in which case all the callbacks must agree to continue the optimization, otherwise the process is terminated and the current embedding is returned.

### Advanced usage

If we want finer control of the optimization process, we can run individual optimization phases (early/late exaggeration) as desired. A typical run of tSNE in scikit-learn using this interface is implemented as follows:

```python
tsne = TSNE()
embedding = tsne.prepare_initial(x)
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
embedding.optimize(n_iter=750, momentum=0.8)
```

Note that all the aspects of optimization can be controlled via the `.optimize` method, see the docs for an extensive list of parameters.


## Future work

- Automatically determine which nearest neighbor/gradient method to use depending on the data set size.

## References

[1] Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.Nov (2008): 2579-2605.

[2] Van Der Maaten, Laurens. "Accelerating t-SNE using tree-based algorithms." The Journal of Machine Learning Research 15.1 (2014): 3221-3245.

[3] Linderman, George C., et al. "Efficient Algorithms for t-distributed Stochastic Neighborhood Embedding." arXiv preprint arXiv:1712.09005 (2017).

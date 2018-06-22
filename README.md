# tSNE

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
from tsne.callbacks import ErrorLogger

tsne = TSNE(callbacks=ErrorLogger(), callbacks_every_iters=50)
```

In this instance, the callback is a callable object, but any function that accepts the following parameters is valid.
```python
def callback(iteration, error, embedding):
	...
```

Callbacks are used to control the optimization i.e. every callback must return a boolean value indicating whether or not to continue the optimization.

Additionally, a list of callbacks can also be passed, in which case all the callbacks must agree to continue the optimization, otherwise the process is terminated and the current embedding is returned.

### Advanced usage

If we want finer control of the optimization process, we can run individual optimization phases (early/late exaggeration) as desired. A typical run of tSNE in scikit-learn using this interface is implemented as follows:

```python
tsne = TSNE()
embedding = tsne.get_initial_embedding_for(x)
embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
embedding.optimize(n_iter=750, momentum=0.8)
```

Note that all the aspects of optimization can be controlled via the `.optimize` method, see the docs for an extensive list of parameters.

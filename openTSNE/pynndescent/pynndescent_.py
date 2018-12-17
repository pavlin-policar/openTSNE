# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

import numba
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array

from openTSNE.pynndescent import distances as dist
from openTSNE.pynndescent.rp_trees import (
    make_euclidean_tree,
    make_angular_tree,
    flatten_tree,
    search_flat_tree,
)
from openTSNE.pynndescent.utils import (
    rejection_sample,
    make_heap,
    heap_push,
    unchecked_heap_push,
    deheap_sort,
    smallest_flagged,
    build_candidates,
    tau_rand,
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_initialisations(dist, dist_args):
    @numba.njit(parallel=True, fastmath=True)
    def init_from_random(n_neighbors, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)
        return

    @numba.njit(parallel=True, fastmath=True)
    def init_from_tree(tree, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = search_flat_tree(
                query_points[i],
                tree.hyperplanes,
                tree.offsets,
                tree.children,
                tree.indices,
                rng_state,
            )

            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)

        return

    return init_from_random, init_from_tree


def initialise_search(
    forest, data, query_points, n_neighbors, init_from_random, init_from_tree, rng_state
):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state)

    return results


def make_initialized_nnd_search(dist, dist_args):
    @numba.njit(parallel=True, fastmath=True)
    def initialized_nnd_search(data, indptr, indices, initialization, query_points):

        for i in numba.prange(query_points.shape[0]):

            tried = set(initialization[0, i])

            while True:

                # Find smallest flagged vertex
                vertex = smallest_flagged(initialization, i)

                if vertex == -1:
                    break
                candidates = indices[indptr[vertex] : indptr[vertex + 1]]
                for j in range(candidates.shape[0]):
                    if (
                        candidates[j] == vertex
                        or candidates[j] == -1
                        or candidates[j] in tried
                    ):
                        continue
                    d = dist(data[candidates[j]], query_points[i], *dist_args)
                    unchecked_heap_push(initialization, i, d, candidates[j], 1)
                    tried.add(candidates[j])

        return initialization

    return initialized_nnd_search


def make_nn_descent(dist, dist_args):
    """Create a numba accelerated version of nearest neighbor descent
    specialised for the given distance metric and metric arguments. Numba
    doesn't support higher order functions directly, but we can instead JIT
    compile the version of NN-descent for any given metric.
    Parameters
    ----------
    dist: function
        A numba JITd distance function which, given two arrays computes a
        dissimilarity between them.
    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.
    Returns
    -------
    A numba JITd function for nearest neighbor descent computation that is
    specialised to the given metric.
    """

    @numba.njit(parallel=True)
    def nn_descent(
        data,
        n_neighbors,
        rng_state,
        max_candidates=50,
        n_iters=10,
        delta=0.001,
        rho=0.5,
        rp_tree_init=True,
        leaf_array=None,
        verbose=False,
    ):
        n_vertices = data.shape[0]

        current_graph = make_heap(data.shape[0], n_neighbors)
        for i in range(data.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                d = dist(data[i], data[indices[j]], *dist_args)
                heap_push(current_graph, i, d, indices[j], 1)
                heap_push(current_graph, indices[j], d, i, 1)

        if rp_tree_init:
            for n in range(leaf_array.shape[0]):
                for i in range(leaf_array.shape[1]):
                    if leaf_array[n, i] < 0:
                        break
                    for j in range(i + 1, leaf_array.shape[1]):
                        if leaf_array[n, j] < 0:
                            break
                        d = dist(
                            data[leaf_array[n, i]], data[leaf_array[n, j]], *dist_args
                        )
                        heap_push(
                            current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
                        )
                        heap_push(
                            current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
                        )

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            candidate_neighbors = build_candidates(
                current_graph, n_vertices, n_neighbors, max_candidates, rng_state
            )

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(candidate_neighbors[0, i, j])
                    if p < 0 or tau_rand(rng_state) < rho:
                        continue
                    for k in range(max_candidates):
                        q = int(candidate_neighbors[0, i, k])
                        if (
                            q < 0
                            or not candidate_neighbors[2, i, j]
                            and not candidate_neighbors[2, i, k]
                        ):
                            continue

                        d = dist(data[p], data[q], *dist_args)
                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * data.shape[0]:
                break

        return deheap_sort(current_graph)

    return nn_descent


def make_heap_initializer(dist, dist_args):
    """Create a numba accelerated version of heap initialization for the
    alternative k-neighbor graph algorithm. This approach builds two heaps
    of neighbors simultaneously, one is a heap used to construct a very
    approximate k-neighbor graph for searching; the other is the
    initialization for searching.

    Parameters
    ----------
    dist: function
        A numba JITd distance function which, given two arrays computes a
        dissimilarity between them.

    dist_args: tuple
        Any extra arguments that need to be passed to the distance function
        beyond the two arrays to be compared.

    Returns
    -------
    A numba JITd function for for heap initialization that is
    specialised to the given metric.
    """

    @numba.njit(parallel=True)
    def initialize_heaps(data, n_neighbors, leaf_array):
        graph_heap = make_heap(data.shape[0], 10)
        search_heap = make_heap(data.shape[0], n_neighbors * 2)
        tried = set([(-1, -1)])
        for n in range(leaf_array.shape[0]):
            for i in range(leaf_array.shape[1]):
                if leaf_array[n, i] < 0:
                    break
                for j in range(i + 1, leaf_array.shape[1]):
                    if leaf_array[n, j] < 0:
                        break
                    if (leaf_array[n, i], leaf_array[n, j]) in tried:
                        continue

                    d = dist(data[leaf_array[n, i]], data[leaf_array[n, j]], *dist_args)
                    unchecked_heap_push(
                        graph_heap, leaf_array[n, i], d, leaf_array[n, j], 1
                    )
                    unchecked_heap_push(
                        graph_heap, leaf_array[n, j], d, leaf_array[n, i], 1
                    )
                    unchecked_heap_push(
                        search_heap, leaf_array[n, i], d, leaf_array[n, j], 1
                    )
                    unchecked_heap_push(
                        search_heap, leaf_array[n, j], d, leaf_array[n, i], 1
                    )
                    tried.add((leaf_array[n, i], leaf_array[n, j]))

        return graph_heap, search_heap

    return initialize_heaps


def degree_prune(graph, max_degree=20):
    """Prune the k-neighbors graph back so that nodes have a maximum
    degree of ``max_degree``.

    Parameters
    ----------
    graph: sparse matrix
        The adjacency matrix of the graph

    max_degree: int (optional, default 20)
        The maximum degree of any node in the pruned graph

    Returns
    -------
    result: sparse matrix
        The pruned graph.
    """

    result = graph.tolil()
    for i, row_data in enumerate(result.data):
        if len(row_data) > max_degree:
            cut_value = np.argsort(row_data)[max_degree]
            row_data = [x if x <= cut_value else 0.0 for x in row_data]
            result.data[i] = row_data
    result = result.tocsr()
    result.eliminate_zeros()
    return result


def prune(graph, prune_level=0, n_neighbors=10):
    """Perform pruning on the graph so that there are fewer edges to
    be followed. In practice this operates in two passes. The first pass
    removes edges such that no node has degree more than ``3 * n_neighbors -
    prune_level``. The second pass builds up a graph out of spanning trees;
    each iteration constructs a minimum panning tree of a graph and then
    removes those edges from the graph. The result is spanning trees that
    take various paths through the graph. All these spanning trees are merged
    into the resulting graph. In practice this prunes away a limited number
    of edges as long as enough iterations are performed. By default we will
    do ``n_neighbors - prune_level``iterations.

    Parameters
    ----------
    graph: sparse matrix
        The adjacency matrix of the graph

    prune_level: int (optional default 0)
        How aggressively to prune the graph, larger values perform more
        aggressive pruning.

    n_neighbors: int (optional 10)
        The number of neighbors of the k-neighbor graph that was constructed.

    Returns
    -------
    result: sparse matrix
        The pruned graph
    """

    max_degree = max(5, 3 * n_neighbors - prune_level)
    n_iters = max(3, n_neighbors - prune_level)
    reduced_graph = degree_prune(graph, max_degree=max_degree)
    result_graph = lil_matrix((graph.shape[0], graph.shape[0])).tocsr()

    for _ in range(n_iters):
        mst = minimum_spanning_tree(reduced_graph)
        result_graph = result_graph.maximum(mst)
        reduced_graph -= mst
        reduced_graph.eliminate_zeros()

    return result_graph


class NNDescent(object):
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array os shape (n_samples, n_features)
        The training data set to find nearest neighbors in.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_neighbors: int (optional, default=15)
        The number of neighbors to use in k-neighbor graph data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_trees: int (optional, default=8)
        This implementation uses random projection forests for initialization
        of searches. This parameter controls the number of trees in that
        forest. A larger number will result in ore accurate neighbor
        computation at the cost of performance.

    leaf_size: int (optional, default=15)
        The maximum number of points in a leaf for the random projection trees.

    pruning_level: int (optional, default=0)
        How aggressively to prune the graph. Higher values perform more
        aggressive pruning, resulting in faster search with lower accuracy.

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        To use the alternative algorithm specify ``'alternative'``.

    max_candidates: int (optional, default=20)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=10)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    rho: float (optional, default=0.5)
        Controls the random sampling of potential candidates in any given
        iteration of NN-descent. Larger values will result in less accurate
        indexes and less accurate searching. Don't tweak this value unless
        you know what you're doing.
    """

    def __init__(
        self,
        data,
        metric="euclidean",
        metric_kwds=None,
        n_neighbors=15,
        n_trees=8,
        leaf_size=15,
        pruning_level=0,
        tree_init=True,
        random_state=np.random,
        algorithm="standard",
        max_candidates=20,
        n_iters=10,
        delta=0.001,
        rho=0.5,
    ):

        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.metric = metric

        if metric_kwds is None:
            metric_kwds = dict()
        self.metric_kwds = metric_kwds

        self.leaf_size = leaf_size
        self.prune_level = pruning_level
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.delta = delta
        self.rho = rho
        self.dim = data.shape[1]

        data = check_array(data).astype(np.float32)

        if not tree_init or n_trees == 0:
            self.tree_init = False
        else:
            self.tree_init = True

        self._dist_args = tuple(metric_kwds.values())

        self.random_state = check_random_state(random_state)

        self._raw_data = data.copy()

        if callable(metric):
            self._distance_func = metric
        elif metric in dist.named_distances:
            self._distance_func = dist.named_distances[metric]

        if metric in ("cosine", "correlation", "dice", "jaccard"):
            self._angular_trees = True
        else:
            self._angular_trees = False

        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )

        indices = np.arange(data.shape[0])

        if self.tree_init:
            if self._angular_trees:
                self._rp_forest = [
                    flatten_tree(
                        make_angular_tree(
                            data, indices, self.rng_state, self.leaf_size
                        ),
                        self.leaf_size,
                    )
                    for i in range(n_trees)
                ]
            else:
                self._rp_forest = [
                    flatten_tree(
                        make_euclidean_tree(
                            data, indices, self.rng_state, self.leaf_size
                        ),
                        self.leaf_size,
                    )
                    for i in range(n_trees)
                ]

            leaf_array = np.vstack([tree.indices for tree in self._rp_forest])
        else:
            self._rp_forest = None
            leaf_array = np.array([[-1]])

        if algorithm == "standard" or leaf_array.shape[0] == 1:
            nn_descent = make_nn_descent(self._distance_func, self._dist_args)
            self._neighbor_graph = nn_descent(
                self._raw_data,
                self.n_neighbors,
                self.rng_state,
                self.max_candidates,
                self.n_iters,
                self.delta,
                self.rho,
                True,
                leaf_array,
            )
        elif algorithm == "alternative":
            self._search = make_initialized_nnd_search(
                self._distance_func, self._dist_args
            )

            init_heaps = make_heap_initializer(self._distance_func, self._dist_args)
            graph_heap, search_heap = init_heaps(
                self._raw_data, self.n_neighbors, leaf_array
            )
            graph = lil_matrix((data.shape[0], data.shape[0]))
            graph.rows, graph.data = deheap_sort(graph_heap)
            graph = graph.maximum(graph.transpose())
            self._neighbor_graph = deheap_sort(
                self._search(
                    self._raw_data,
                    graph.indptr,
                    graph.indices,
                    search_heap,
                    self._raw_data,
                )
            )
        else:
            raise ValueError("Unknown algorithm selected")

        self._search_graph = lil_matrix(
            (data.shape[0], data.shape[0]), dtype=np.float32
        )
        self._search_graph.rows = self._neighbor_graph[0]
        self._search_graph.data = self._neighbor_graph[1]
        self._search_graph = self._search_graph.maximum(
            self._search_graph.transpose()
        ).tocsr()
        self._search_graph = prune(
            self._search_graph,
            prune_level=self.prune_level,
            n_neighbors=self.n_neighbors,
        )
        self._search_graph = (self._search_graph != 0).astype(np.int8)

        self._random_init, self._tree_init = make_initialisations(
            self._distance_func, self._dist_args
        )

        self._search = make_initialized_nnd_search(self._distance_func, self._dist_args)

        return

    def query(self, query_data, k=10, queue_size=5.0):
        """Query the training data for the k nearest neighbors

        Parameters
        ----------
        query_data: array-like, last dimension self.dim
            An array of points to query

        k: integer (default = 10)
            The number of nearest neighbors to return

        queue_size: float (default 5.0)
            The multiplier of the internal search queue. This controls the
            speed/accuracy tradeoff. Low values will search faster but with
            more approximate results. High values will search more
            accurately, but will require more computation to do so. Values
            should generally be in the range 1.0 to 10.0.

        Returns
        -------
        indices, distances: array (n_query_points, k), array (n_query_points, k)
            The first array, ``indices``, provides the indices of the data
            points in the training set that are the nearest neighbors of
            each query point. Thus ``indices[i, j]`` is the index into the
            training data of the jth nearest neighbor of the ith query points.

            Similarly ``distances`` provides the distances to the neighbors
            of the query points such that ``distances[i, j]`` is the distance
            from the ith query point to its jth nearest neighbor in the
            training data.
        """
        # query_data = check_array(query_data, dtype=np.float64, order='C')
        query_data = np.asarray(query_data).astype(np.float32)
        init = initialise_search(
            self._rp_forest,
            self._raw_data,
            query_data,
            int(k * queue_size),
            self._random_init,
            self._tree_init,
            self.rng_state,
        )
        result = self._search(
            self._raw_data,
            self._search_graph.indptr,
            self._search_graph.indices,
            init,
            query_data,
        )

        indices, dists = deheap_sort(result)
        return indices[:, :k], dists[:, :k]


class PyNNDescentTransformer(BaseEstimator, TransformerMixin):
    """PyNNDescentTransformer for fast approximate nearest neighbor transformer.
    It uses the NNDescent algorithm, and is thus
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    data in many cases.

    Transform X into a (weighted) graph of k nearest neighbors

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Parameters
    ----------
    n_neighbors: int (optional, default=5)
        The number of neighbors to use in k-neighbor graph data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    metric: string or callable (optional, default='euclidean')
        The metric to use for computing nearest neighbors. If a callable is
        used it must be a numba njit compiled function. Supported metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    n_trees: int (optional, default=8)
        This implementation uses random projection forests for initialization
        of searches. This parameter controls the number of trees in that
        forest. A larger number will result in ore accurate neighbor
        computation at the cost of performance.

    leaf_size: int (optional, default=15)
        The maximum number of points in a leaf for the random projection trees.

    pruning_level: int (optional, default=0)
        How aggressively to prune the graph. Higher values perform more
        aggressive pruning, resulting in faster search with lower accuracy.

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        To use the alternative algorithm specify ``'alternative'``.

    max_candidates: int (optional, default=20)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=10)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    rho: float (optional, default=0.5)
        Controls the random sampling of potential candidates in any given
        iteration of NN-descent. Larger values will result in less accurate
        indexes and less accurate searching. Don't tweak this value unless
        you know what you're doing.

    Examples
    --------
    >>> from sklearn.manifold import Isomap
    >>> from pynndescent import PyNNDescentTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> estimator = make_pipeline(
    ...     PyNNDescentTransformer(n_neighbors=5),
    ...     Isomap(neighbors_algorithm='precomputed'))
    """

    def __init__(
        self,
        n_neighbors=5,
        metric="euclidean",
        metric_kwds=None,
        n_trees=8,
        leaf_size=15,
        search_queue_size=4.0,
        pruning_level=0,
        tree_init=True,
        random_state=np.random,
        algorithm="standard",
        max_candidates=20,
        n_iters=10,
        early_termination_value=0.001,
        sampling_rate=0.5,
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.search_queue_size = search_queue_size
        self.pruning_level = pruning_level
        self.tree_init = tree_init
        self.random_state = random_state
        self.algorithm = algorithm
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.early_termination_value = early_termination_value
        self.sampling_rate = sampling_rate

    def fit(self, X):
        """Fit the PyNNDescent transformer to build KNN graphs with
        neighbors given by the dataset X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample data

        Returns
        -------
        transformer : PyNNDescentTransformer
            The trained transformer
        """
        self.n_samples_fit = X.shape[0]

        if self.metric_kwds is None:
            metric_kwds = {}
        else:
            metric_kwds = self.metric_kwds

        self.pynndescent_ = NNDescent(
            X,
            self.metric,
            metric_kwds,
            self.n_neighbors,
            self.n_trees,
            self.leaf_size,
            self.pruning_level,
            self.tree_init,
            self.random_state,
            self.algorithm,
            self.max_candidates,
            self.n_iters,
            self.early_termination_value,
            self.sampling_rate,
        )

        return self

    def transform(self, X, y=None):
        """Computes the (weighted) graph of Neighbors for points in X

        Parameters
        ----------
        X : array-like, shape (n_samples_transform, n_features)
            Sample data

        Returns
        -------
        Xt : CSR sparse matrix, shape (n_samples_fit, n_samples_transform)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
        """

        if X is None:
            n_samples_transform = self.n_samples_fit
        else:
            n_samples_transform = X.shape[0]

        if X is None:
            indices, distances = self.pynndescent_._neighbor_graph
        else:
            indices, distances = self.pynndescent_.query(
                X, k=self.n_neighbors, queue_size=self.search_queue_size
            )

        result = lil_matrix(
            (n_samples_transform, n_samples_transform), dtype=np.float32
        )
        result.rows = indices
        result.data = distances

        return result.tocsr()

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training set.

        y : ignored

        Returns
        -------
        Xt : CSR sparse matrix, shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
        """
        return self.fit(X).transform(X=None)

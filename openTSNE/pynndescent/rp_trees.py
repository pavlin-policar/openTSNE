# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause
import numba
import numpy as np

from collections import namedtuple

from openTSNE.pynndescent.utils import tau_rand_int, norm


RandomProjectionTreeNode = namedtuple(
    "RandomProjectionTreeNode",
    ["indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
)

FlatTree = namedtuple("FlatTree", ["hyperplanes", "offsets", "children", "indices"])


@numba.njit(fastmath=True, nogil=True, parallel=True)
def euclidean_random_projection_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses euclidean distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split

    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.

    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if margin == 0:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, hyperplane_offset


@numba.njit()
def angular_random_projection_split(data, indices, rng_state):
    """Given a set of ``indices`` for data points from ``data``, create
    a random hyperplane to split the data, returning two arrays indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.

    This particular split uses cosine distance to determine the hyperplane
    and which side each data sample falls on.

    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original data to be split

    indices: array of shape (tree_node_size,)
        The indices of the elements in the ``data`` array that are to
        be split in the current operation.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    indices_left: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.

    indices_right: array
        The elements of ``indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = norm(data[left])
    right_norm = norm(data[right])

    if left_norm == 0.0:
        left_norm = 1.0

    if right_norm == 0.0:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if hyperplane_norm == 0.0:
        hyperplane_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if margin == 0:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    # Populate the arrays with indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, 0


def make_euclidean_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = euclidean_random_projection_split(
            data, indices, rng_state
        )

        left_node = make_euclidean_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_euclidean_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_angular_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = angular_random_projection_split(
            data, indices, rng_state
        )

        left_node = make_angular_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_angular_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def num_nodes(tree):
    """Determine the number of nodes in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return 1 + num_nodes(tree.left_child) + num_nodes(tree.right_child)


def num_leaves(tree):
    """Determine the number of leaves in a tree"""
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)


def recursive_flatten(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_num
):
    if tree.is_leaf:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : tree.indices.shape[0]] = tree.indices
        leaf_num += 1
        return node_num, leaf_num
    else:
        hyperplanes[node_num] = tree.hyperplane
        offsets[node_num] = tree.offset
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_flatten(
            tree.left_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_flatten(
            tree.right_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        return node_num, leaf_num


def flatten_tree(tree, leaf_size):
    n_nodes = num_nodes(tree)
    n_leaves = num_leaves(tree)
    hyperplanes = np.zeros((n_nodes, tree.hyperplane.shape[0]), dtype=np.float32)
    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
    indices = -1 * np.ones((n_leaves, leaf_size), dtype=np.int64)
    recursive_flatten(tree, hyperplanes, offsets, children, indices, 0, 0)
    return FlatTree(hyperplanes, offsets, children, indices)


@numba.njit(fastmath=True)
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    for d in range(point.shape[0]):
        margin += hyperplane[d] * point[d]

    if margin == 0:
        side = tau_rand_int(rng_state) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(fastmath=True)
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]

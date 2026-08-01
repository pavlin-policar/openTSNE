# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
"""Implements a quad/oct-tree space partitioning algorithm primarily used in
efficiently estimating the t-SNE negative gradient. Lowers the time complexity
from the naive O(n^2) to O(n * log(n)).

Notes
-----
I list here several implementation details. Many of these improve efficiency.

  - Allocating memory is slow, especially if it has to be done millions of
    times, therefore avoid allocation whereever possible and use buffers.
    Allocation should be done through the use of `PyMem_Malloc` as this is the
    fastest method of allocation. Use this over `libc.stdlib.malloc` because,
    despite requiring the GIL to allocate, it gets tracked in the Python
    virtual environment (which is desirable) and includes some minor
    optimizations. Also, since we need the GIL to allocate, this can warn us of
    any needless memory allocations.

  - Structs do not support memoryviews, therefore pointers must be used.

  - Prefer pointers over memoryviews where speed is essential. Memoryview
    indexing and slicing is slow compared to raw memory access. We can easily
    convert a memory view to a pointer like so: `&mv[0]` however care must be
    taken to ensure the memoryview is a C contigous array. This can be ensured
    by the type declaration `double[:, ::1]` for 2d arrays.

References
----------
.. [1] Van Der Maaten, Laurens. "Accelerating t-SNE using tree-based
   algorithms." Journal of machine learning research 15.1 (2014): 3221-3245.

"""
cimport numpy as cnp
cnp.import_array()
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from "math.h":
    double fabs(double x) nogil


cdef void init_node(Node * node, Py_ssize_t n_dim, double * center, double length):
    node.n_dims = n_dim
    node.center = <double *>PyMem_Malloc(node.n_dims * sizeof(double))
    node.center_of_mass = <double *>PyMem_Malloc(node.n_dims * sizeof(double))
    if not node.center or not node.center_of_mass:
        raise MemoryError()

    cdef Py_ssize_t i
    for i in range(node.n_dims):
        node.center[i] = center[i]
        node.center_of_mass[i] = 0

    node.length = length

    node.is_leaf = True
    node.num_points = 0


cdef Py_ssize_t get_child_idx_for(Node * node, double * point) noexcept nogil:
    cdef Py_ssize_t idx = 0, d

    for d in range(node.n_dims):
        idx |= (point[d] > node.center[d]) << d

    return idx


cdef inline void update_center_of_mass(Node * node, double * point) noexcept nogil:
    cdef Py_ssize_t d
    for d in range(node.n_dims):
        node.center_of_mass[d] = (node.center_of_mass[d] * node.num_points + point[d]) / (node.num_points + 1)
    node.num_points += 1


cdef void add_point_to(Node * node, double * point):
    # The point must lie inside the node's bounding box. The descent terminates
    # because each child box is half the size of its parent and still contains
    # the point, so the box eventually shrinks around it. For a point outside
    # the box, every child is equally far from it and the descent never ends.

    # If the node is a leaf node and empty, we're done. We'll also limit the
    # branching of each node to prevent memory explosions. 1e-6 here is
    # effectively the minimum size that a node can take
    if node.is_leaf and node.num_points == 0 or is_close(node, point, 1e-6):
        update_center_of_mass(node, point)
        return

    # Otherwise, we have to split the node and sink the previous, existing
    # point into the appropriate child node
    cdef Py_ssize_t child_index

    if node.is_leaf:
        split_node(node)
        child_index = get_child_idx_for(node, node.center_of_mass)
        update_center_of_mass(&node.children[child_index], node.center_of_mass)

    update_center_of_mass(node, point)

    # Finally, once the node is properly split, insert the new point into the
    # corresponding child
    child_index = get_child_idx_for(node, point)
    add_point_to(&node.children[child_index], point)


cdef void split_node(Node * node):
    cdef double new_length = node.length / 2
    cdef Py_ssize_t num_children = 1 << node.n_dims

    node.is_leaf = False
    node.children = <Node *>PyMem_Malloc(num_children * sizeof(Node))
    if not node.children:
        raise MemoryError()

    cdef Py_ssize_t i, d
    cdef double * new_center = <double *>PyMem_Malloc(node.n_dims * sizeof(double))
    if not new_center:
        raise MemoryError()

    for i in range(num_children):
        for d in range(node.n_dims):
            if i & (1 << d):
                new_center[d] = node.center[d] + new_length / 2
            else:
                new_center[d] = node.center[d] - new_length / 2
        init_node(&node.children[i], node.n_dims, new_center, new_length)

    PyMem_Free(new_center)


cdef inline bint is_close(Node * node, double * point, double eps) noexcept nogil:
    cdef Py_ssize_t d
    for d in range(node.n_dims):
        if fabs(node.center_of_mass[d] - point[d]) >= eps:
            return False
    return True


cdef inline bint is_in_bounds(Node * node, double * point) noexcept nogil:
    cdef Py_ssize_t d
    cdef double half_length = node.length / 2
    for d in range(node.n_dims):
        if not node.center[d] - half_length <= point[d] <= node.center[d] + half_length:
            return False
    return True


cdef raise_out_of_bounds(Node * node, double * point):
    cdef Py_ssize_t d
    cdef double half_length = node.length / 2
    raise ValueError(
        "Point %s lies outside the bounding box of the tree, which spans %s to "
        "%s. The bounding box is fixed at construction time by the data the tree "
        "was built from, and cannot grow to accommodate new points."
        % (
            [point[d] for d in range(node.n_dims)],
            [node.center[d] - half_length for d in range(node.n_dims)],
            [node.center[d] + half_length for d in range(node.n_dims)],
        )
    )


cdef void delete_node(Node * node):
    PyMem_Free(node.center)
    PyMem_Free(node.center_of_mass)
    if node.is_leaf:
        return

    cdef Py_ssize_t i
    for i in range(1 << node.n_dims):
        delete_node(&node.children[i])
    PyMem_Free(node.children)


cdef class QuadTree:
    def __init__(self, double[:, ::1] data):
        cdef:
            Py_ssize_t n_dim = data.shape[1]
            double[:] x_min = np.min(data, axis=0)
            double[:] x_max = np.max(data, axis=0)

            double[:] center = np.zeros(n_dim)
            double length = 0
            Py_ssize_t d, i

        for d in range(n_dim):
            center[d] = (x_max[d] + x_min[d]) / 2
            if x_max[d] - x_min[d] > length:
                length = x_max[d] - x_min[d]

        self.root = Node()
        init_node(&self.root, n_dim, &center[0], length)

        # These points define the bounding box, so they bypass the bounds check
        # the public insertion methods apply. Rounding in the center and side
        # length above can leave an extreme point a fraction of an ulp outside
        # the very box it defines.
        for i in range(data.shape[0]):
            add_point_to(&self.root, &data[i, 0])

    cpdef void add_points(self, double[:, ::1] points):
        """Insert points into the tree.

        Raises
        ------
        ValueError
            If any point lies outside the tree's bounding box. No points are
            inserted in that case.

        """
        cdef Py_ssize_t i
        for i in range(points.shape[0]):
            if not is_in_bounds(&self.root, &points[i, 0]):
                raise_out_of_bounds(&self.root, &points[i, 0])

        for i in range(points.shape[0]):
            add_point_to(&self.root, &points[i, 0])

    cpdef void add_point(self, double[::1] point):
        """Insert a single point into the tree.

        Raises
        ------
        ValueError
            If the point lies outside the tree's bounding box.

        """
        if not is_in_bounds(&self.root, &point[0]):
            raise_out_of_bounds(&self.root, &point[0])
        add_point_to(&self.root, &point[0])

    def __dealloc__(self):
        delete_node(&self.root)

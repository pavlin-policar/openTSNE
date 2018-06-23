# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np

cdef double EPSILON = np.finfo(np.float64).eps

ctypedef struct Node:
    Py_ssize_t n_dims
    double *center
    double length

    bint is_leaf
    Node *children

    double *center_of_mass
    Py_ssize_t num_points


cdef bint is_duplicate(Node * node, double * point, double duplicate_eps=*) nogil


cdef class QuadTree:
    cdef Node root
    cpdef void add_points(self, double[:, ::1] points)
    cpdef void add_point(self, double[::1] point)

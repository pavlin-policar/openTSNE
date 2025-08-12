# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()

ctypedef struct Node:
    Py_ssize_t n_dims
    double *center
    double length

    bint is_leaf
    Node *children

    double *center_of_mass
    Py_ssize_t num_points


cdef bint is_close(Node * node, double * point, double eps) noexcept nogil


cdef class QuadTree:
    cdef Node root
    cpdef void add_points(self, double[:, ::1] points)
    cpdef void add_point(self, double[::1] point)

# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
from libcpp.vector cimport vector
from cython.parallel import prange
import numpy as np
cimport numpy as np


cdef extern from "vptree.h":
    cdef cppclass DataPoint:
        DataPoint() nogil except +
        DataPoint(int, int, double*) nogil except +
        DataPoint(DataPoint&) nogil
        int index() nogil

    cdef cppclass VpTree[T]:
        VpTree() nogil except +
        void create(vector[T]&) nogil
        void search(T, int, vector[T]*, vector[double]*) nogil


cdef class VPTree:
    cdef VpTree[DataPoint]* tree

    def __cinit__(self):
        self.tree = new VpTree[DataPoint]()

    def __dealloc__(self):
        del self.tree

    def __init__(self, double[:, ::1] data):
        self.create(data)

    cdef create(self, double[:, ::1] data):
        # Put the data into a vector of ``DataPoint`` instances
        cdef vector[DataPoint] data_points
        cdef Py_ssize_t idx, N = data.shape[0], n_dim = data.shape[1]

        for idx in range(N):
            data_points.push_back(DataPoint(n_dim, idx, &data[idx, 0]))

        self.tree.create(data_points)

    def search(self, double[:, ::1] query, int K, Py_ssize_t num_threads=1):
        cdef Py_ssize_t i, j, N = query.shape[0], n_dim = query.shape[1]
        # Define objects to be returned to python
        cdef Py_ssize_t[:, ::1] indices = np.empty((N, K), dtype=np.int64)
        cdef double[:, ::1] distances = np.empty((N, K), dtype=np.float64)
        # Define objects to be used internally in vptree
        cdef vector[DataPoint]* indices_
        cdef vector[double]* distances_
        cdef DataPoint query_point

        if num_threads < 1:
            num_threads = 1

        for i in prange(N, nogil=True, schedule="guided", num_threads=num_threads):
            query_point = DataPoint(n_dim, i, &query[i, 0])
            indices_ = new vector[DataPoint]()
            distances_ = new vector[double]()

            self.tree.search(query_point, K, indices_, distances_)

            for j in range(K):
                indices[i, j] = indices_.at(j).index()
                distances[i, j] = distances_.at(j)

            del indices_, distances_

        return np.asarray(indices), np.asarray(distances)

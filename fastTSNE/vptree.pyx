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

cdef extern from *:
    ctypedef int euclidean_distance_t "euclidean_distance"

cdef extern from "vptree.h":
    cdef cppclass DataPoint:
        DataPoint() nogil except +
        DataPoint(int, int, double*) nogil except +
        DataPoint(DataPoint&) nogil
        int index() nogil
        int dimensionality() nogil

    cdef cppclass VpTree[T, F]:
        VpTree() nogil except +
        void create(vector[T]&) nogil
        void search(T&, int, vector[T]*, vector[double]*) nogil


cdef class VPTree:
    cdef VpTree[DataPoint, euclidean_distance_t]* tree
    cdef vector[DataPoint] train_data
    cdef double[:, ::1] data

    valid_metrics = ["euclidean"]

    def __cinit__(self):
        self.tree = new VpTree[DataPoint, euclidean_distance_t]()

    def __dealloc__(self):
        del self.tree

    def __init__(self, double[:, ::1] data):
        # It is important to keep a reference to ``data`` because otherwise
        # (this is an educated guess), the ``*data`` field in DataPoint is
        # dereferenced during execution, resulting in segfaults.
        self.data = data
        self.create(data)

    cdef create(self, double[:, ::1] data):
        # Put the data into a vector of ``DataPoint`` instances
        self.train_data.clear()
        cdef Py_ssize_t idx, N = data.shape[0], n_dim = data.shape[1]

        for idx in range(N):
            self.train_data.push_back(DataPoint(n_dim, idx, &data[idx, 0]))

        self.tree.create(self.train_data)

    def query_train(self, int k, Py_ssize_t num_threads=1):
        # Define objects to be returned
        cdef Py_ssize_t N = self.train_data.size()
        cdef np.int64_t[:, ::1] indices = np.empty((N, k), dtype=np.int64)
        cdef double[:, ::1] distances = np.empty((N, k), dtype=np.float64)

        self.__query(self.train_data, k, indices, distances, num_threads)

        return np.asarray(indices, dtype=np.int64), np.asarray(distances, np.float64)

    def query(self, double[:, ::1] query, int k, Py_ssize_t num_threads=1):
        cdef vector[DataPoint] query_points
        cdef Py_ssize_t idx, N = query.shape[0], n_dim = query.shape[1]

        # Define objects to be returned
        cdef np.int64_t[:, ::1] indices = np.empty((N, k), dtype=np.int64)
        cdef double[:, ::1] distances = np.empty((N, k), dtype=np.float64)

        for idx in range(N):
            query_points.push_back(DataPoint(n_dim, idx, &query[idx, 0]))

        self.__query(query_points, k, indices, distances, num_threads)

        return np.asarray(indices, dtype=np.int64), np.asarray(distances, np.float64)

    cdef __query(
            self,
            vector[DataPoint] query,
            int k,
            np.int64_t[:, ::1] indices,
            double[:, ::1] distances,
            Py_ssize_t num_threads=1,
    ):
        cdef Py_ssize_t i, j, N = query.size(), n_dim = query.at(0).dimensionality()
        # Define objects to be used internally in vptree
        cdef vector[DataPoint]* indices_
        cdef vector[double]* distances_

        if num_threads < 1:
            num_threads = 1

        # for i in prange(N, nogil=True, schedule="guided", num_threads=num_threads):
        for i in range(N):
            indices_ = new vector[DataPoint]()
            distances_ = new vector[double]()

            self.tree.search(query.at(i), k, indices_, distances_)

            for j in range(k):
                indices[i, j] = indices_.at(j).index()
                distances[i, j] = distances_.at(j)

            del indices_, distances_

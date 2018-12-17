# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
from .quad_tree cimport QuadTree


cpdef double[:, ::1] compute_gaussian_perplexity(
    double[:, :] distances,
    double[:] desired_perplexities,
    double perplexity_tol=*,
    Py_ssize_t max_iter=*,
    Py_ssize_t num_threads=*,
)

cpdef tuple estimate_positive_gradient_nn(
    int[:] indices,
    int[:] indptr,
    double[:] P_data,
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    double dof=*,
    Py_ssize_t num_threads=*,
    bint should_eval_error=*,
)

cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=*,
    double dof=*,
    Py_ssize_t num_threads=*,
    bint pairwise_normalization=*,
)

cpdef double estimate_negative_gradient_fft_1d(
    double[::1] embedding,
    double[::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
)

cpdef double estimate_negative_gradient_fft_1d_with_reference(
    double[::1] embedding,
    double[::1] reference_embedding,
    double[::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
)

cpdef double estimate_negative_gradient_fft_2d(
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
)

cpdef double estimate_negative_gradient_fft_2d_with_reference(
    double[:, ::1] embedding,
    double[:, ::1] reference_embedding,
    double[:, ::1] gradient,
    Py_ssize_t n_interpolation_points=*,
    Py_ssize_t min_num_intervals=*,
    double ints_in_interval=*,
)

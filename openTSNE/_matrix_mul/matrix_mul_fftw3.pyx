# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as cnp
cnp.import_array()
import threading

import numpy as np
from cython.parallel cimport prange
from libc.string cimport memcpy, memset
cimport openTSNE._matrix_mul.matrix_mul


cdef extern from 'fftw3.h':
    cdef int FFTW_FORWARD
    cdef int FFTW_BACKWARD
    cdef unsigned FFTW_ESTIMATE
    cdef unsigned FFTW_DESTROY_INPUT
    ctypedef double fftw_complex[2]

    ctypedef struct _fftw_plan:
       pass

    ctypedef _fftw_plan *fftw_plan

    void fftw_execute(fftw_plan) nogil
    void fftw_execute_dft(fftw_plan, fftw_complex *, fftw_complex *) nogil
    void fftw_execute_dft_r2c(fftw_plan, double *, fftw_complex *) nogil
    void fftw_execute_dft_c2r(fftw_plan, fftw_complex *, double *) nogil
    void fftw_destroy_plan(fftw_plan)
    fftw_plan fftw_plan_dft_r2c_1d(int, double*, fftw_complex*, unsigned)
    fftw_plan fftw_plan_dft_c2r_1d(int, fftw_complex*, double*, unsigned)
    fftw_plan fftw_plan_many_dft(
        int rank, const int *n, int howmany,
        fftw_complex *in_, const int *inembed, int istride, int idist,
        fftw_complex *out, const int *onembed, int ostride, int odist,
        int sign, unsigned flags,
    )
    fftw_plan fftw_plan_many_dft_r2c(
        int rank, const int *n, int howmany,
        double *in_, const int *inembed, int istride, int idist,
        fftw_complex *out, const int *onembed, int ostride, int odist,
        unsigned flags,
    )
    fftw_plan fftw_plan_many_dft_c2r(
        int rank, const int *n, int howmany,
        fftw_complex *in_, const int *inembed, int istride, int idist,
        double *out, const int *onembed, int ostride, int odist,
        unsigned flags,
    )


# Only FFTW's execute functions are safe to call concurrently; creating and
# destroying plans is not, and embeddings optimized in different threads plan
# independently. Reentrant, since a plan's owner can be garbage collected, and
# its plans destroyed, while the same thread holds the lock.
cdef object _planner_lock = threading.RLock()

# Plans and their buffers are tied to one transform size. The grid size changes
# only when the embedding's extent crosses a box boundary, so each thread keeps
# the plans for the last size it used; holding on to more would waste more
# memory than re-planning costs. Each thread keeps its own because plans write
# into their buffers, and embeddings optimized concurrently in different
# threads must not share them.
cdef object _local = threading.local()


cdef void _destroy_plan(fftw_plan plan):
    if plan == NULL:
        return
    # The lock may already be gone when plans are freed at interpreter exit,
    # when no other thread can be planning
    if _planner_lock is None:
        fftw_destroy_plan(plan)
        return
    with _planner_lock:
        fftw_destroy_plan(plan)


cdef inline double complex *_complex_data(cnp.ndarray arr):
    return <double complex *>cnp.PyArray_DATA(arr)


cdef inline double *_real_data(cnp.ndarray arr):
    return <double *>cnp.PyArray_DATA(arr)


cdef class _Plans1D:
    """Plans for the 1d circulant product of length ``n``.

    The 1d transforms are at most a few thousand points long, too short for
    threads to pay off.
    """
    cdef:
        readonly Py_ssize_t n
        cnp.ndarray kernel_in, kernel_spec, signal, spec, result
        fftw_plan kernel, forward, backward

    def __cinit__(self, Py_ssize_t n):
        self.n = n
        self.kernel_in = np.zeros(n, dtype=float)
        self.kernel_spec = np.zeros(n // 2 + 1, dtype=complex)
        # Only the leading half is ever written, so the rest stays zero
        self.signal = np.zeros(n, dtype=float)
        self.spec = np.zeros(n // 2 + 1, dtype=complex)
        self.result = np.zeros(n, dtype=float)

        with _planner_lock:
            self.kernel = fftw_plan_dft_r2c_1d(
                n, _real_data(self.kernel_in),
                <fftw_complex *>_complex_data(self.kernel_spec), FFTW_ESTIMATE,
            )
            self.forward = fftw_plan_dft_r2c_1d(
                n, _real_data(self.signal),
                <fftw_complex *>_complex_data(self.spec), FFTW_ESTIMATE,
            )
            self.backward = fftw_plan_dft_c2r_1d(
                n, <fftw_complex *>_complex_data(self.spec),
                _real_data(self.result), FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
            )
        if self.kernel == NULL or self.forward == NULL or self.backward == NULL:
            raise MemoryError("FFTW could not create a plan of size %d" % n)

    def __dealloc__(self):
        _destroy_plan(self.kernel)
        _destroy_plan(self.forward)
        _destroy_plan(self.backward)


# The kinds of batched 1d transform a pass consists of
cdef enum PassKind:
    REAL_TO_COMPLEX
    COMPLEX_TO_REAL
    COMPLEX_FORWARD
    COMPLEX_BACKWARD


cdef class _Pass:
    """A batch of ``count`` independent 1d transforms of length ``n``, split
    into chunks of ``chunk`` transforms that can run on different threads.

    Transform ``i`` reads from ``in_ + i * in_dist`` and writes to
    ``out + i * out_dist``, with ``stride`` between consecutive elements of
    one transform. Every chunk runs the same single-threaded plan through
    FFTW's new-array execute functions, which are safe to call concurrently;
    the last chunk has its own plan when it is shorter. All chunks start a
    whole number of elements apart, so they share the plan's alignment.
    """
    cdef:
        PassKind kind
        Py_ssize_t count, chunk, n_chunks
        # Distances between consecutive transforms, in bytes
        Py_ssize_t in_step, out_step
        char *in_
        char *out
        fftw_plan full, last

    @staticmethod
    cdef _Pass create(
        PassKind kind, int n, Py_ssize_t count, Py_ssize_t chunk,
        void *in_, int in_dist, void *out, int out_dist, int stride,
    ):
        cdef:
            _Pass p = _Pass.__new__(_Pass)
            Py_ssize_t in_size = sizeof(double) if kind == REAL_TO_COMPLEX else sizeof(double complex)
            Py_ssize_t out_size = sizeof(double) if kind == COMPLEX_TO_REAL else sizeof(double complex)

        p.kind = kind
        p.count = count
        p.chunk = min(chunk, count)
        p.n_chunks = (count + p.chunk - 1) // p.chunk
        p.in_ = <char *>in_
        p.out = <char *>out
        p.in_step = in_dist * in_size
        p.out_step = out_dist * out_size

        with _planner_lock:
            p.full = p._plan(n, p.chunk, in_dist, out_dist, stride)
            if count % p.chunk:
                p.last = p._plan(n, count % p.chunk, in_dist, out_dist, stride)
        if p.full == NULL or (count % p.chunk and p.last == NULL):
            raise MemoryError("FFTW could not create a plan of size %d" % n)
        return p

    cdef fftw_plan _plan(self, int n, int howmany, int in_dist, int out_dist, int stride):
        if self.kind == REAL_TO_COMPLEX:
            return fftw_plan_many_dft_r2c(
                1, &n, howmany,
                <double *>self.in_, NULL, stride, in_dist,
                <fftw_complex *>self.out, NULL, stride, out_dist,
                FFTW_ESTIMATE,
            )
        if self.kind == COMPLEX_TO_REAL:
            return fftw_plan_many_dft_c2r(
                1, &n, howmany,
                <fftw_complex *>self.in_, NULL, stride, in_dist,
                <double *>self.out, NULL, stride, out_dist,
                FFTW_ESTIMATE | FFTW_DESTROY_INPUT,
            )
        return fftw_plan_many_dft(
            1, &n, howmany,
            <fftw_complex *>self.in_, NULL, stride, in_dist,
            <fftw_complex *>self.out, NULL, stride, out_dist,
            FFTW_FORWARD if self.kind == COMPLEX_FORWARD else FFTW_BACKWARD,
            FFTW_ESTIMATE,
        )

    cdef void run(self, Py_ssize_t num_threads) noexcept nogil:
        cdef:
            Py_ssize_t c
            fftw_plan plan
            char *in_
            char *out

        if self.n_chunks == 1:
            self._execute(self.full, self.in_, self.out)
            return

        for c in prange(self.n_chunks, num_threads=num_threads, schedule="dynamic"):
            plan = self.last if (c == self.n_chunks - 1 and self.last != NULL) else self.full
            in_ = self.in_ + c * self.chunk * self.in_step
            out = self.out + c * self.chunk * self.out_step
            self._execute(plan, in_, out)

    cdef inline void _execute(self, fftw_plan plan, char *in_, char *out) noexcept nogil:
        if self.kind == REAL_TO_COMPLEX:
            fftw_execute_dft_r2c(plan, <double *>in_, <fftw_complex *>out)
        elif self.kind == COMPLEX_TO_REAL:
            fftw_execute_dft_c2r(plan, <fftw_complex *>in_, <double *>out)
        else:
            fftw_execute_dft(plan, <fftw_complex *>in_, <fftw_complex *>out)

    def __dealloc__(self):
        _destroy_plan(self.full)
        _destroy_plan(self.last)


# A parallel region costs more than the transforms it splits unless each
# thread gets enough of them; small grids run on fewer threads
cdef enum:
    MIN_TRANSFORMS_PER_THREAD = 64


cdef inline Py_ssize_t _chunk_size(Py_ssize_t count, Py_ssize_t num_threads):
    """Transforms per chunk: several chunks per thread, so that a thread on a
    slower or busier core holds up the others less, but enough transforms per
    chunk for FFTW to vectorize across them."""
    if num_threads <= 1:
        return count
    return max(8, (count + 4 * num_threads - 1) // (4 * num_threads))


cdef class _Plans2D:
    """Plans for the 2d circulant product on an ``n x n`` grid.

    A 2d transform is a pass along the rows followed by a pass along the
    columns. Each pass is a batch of independent 1d transforms, which split
    across threads without any coordination, and taking the passes separately
    also lets them skip work that a single 2d plan cannot know about:

      - the coefficients occupy only the top left quadrant of the circulant
        embedding, so the forward row pass is needed for only the top half of
        the rows -- the bottom half is zero;
      - only the bottom right quadrant of the result is ever read, so the
        inverse row pass is needed for only the bottom half of the rows.

    Together that removes about a quarter of the transform work.
    """
    cdef:
        readonly Py_ssize_t n, num_threads
        # The kernel and its full 2d spectrum, (n, n // 2 + 1)
        cnp.ndarray kernel_in, kernel_spec
        # The top half of the zero-padded coefficients, (n // 2, n); only the
        # left half of each row is ever written, so the rest stays zero
        cnp.ndarray rows_in
        # The coefficients' spectrum, (n, n // 2 + 1), transformed in place
        cnp.ndarray spec
        # The bottom half of the product's rows, (n // 2, n)
        cnp.ndarray rows_out
        double *kernel_in_ptr
        double complex *kernel_spec_ptr
        double *rows_in_ptr
        double complex *spec_ptr
        double *rows_out_ptr
        _Pass kernel_rows, kernel_columns
        _Pass row_forward, column_forward, column_backward, row_backward

    def __cinit__(self, Py_ssize_t n, Py_ssize_t num_threads):
        cdef:
            Py_ssize_t half = n // 2
            Py_ssize_t n_cols = n // 2 + 1
            Py_ssize_t row_chunk = _chunk_size(half, num_threads)
            Py_ssize_t column_chunk = _chunk_size(n_cols, num_threads)

        self.n = n
        self.num_threads = num_threads
        self.kernel_in = np.zeros((n, n), dtype=float)
        self.kernel_spec = np.zeros((n, n_cols), dtype=complex)
        self.rows_in = np.zeros((half, n), dtype=float)
        self.spec = np.zeros((n, n_cols), dtype=complex)
        self.rows_out = np.zeros((half, n), dtype=float)
        self.kernel_in_ptr = _real_data(self.kernel_in)
        self.kernel_spec_ptr = _complex_data(self.kernel_spec)
        self.rows_in_ptr = _real_data(self.rows_in)
        self.spec_ptr = _complex_data(self.spec)
        self.rows_out_ptr = _real_data(self.rows_out)

        # Rows are contiguous transforms of length n, one row apart; columns
        # are transforms strided by a row, one element apart
        self.kernel_rows = _Pass.create(
            REAL_TO_COMPLEX, n, n, _chunk_size(n, num_threads),
            self.kernel_in_ptr, n, self.kernel_spec_ptr, n_cols, 1,
        )
        self.kernel_columns = _Pass.create(
            COMPLEX_FORWARD, n, n_cols, column_chunk,
            self.kernel_spec_ptr, 1, self.kernel_spec_ptr, 1, n_cols,
        )
        self.row_forward = _Pass.create(
            REAL_TO_COMPLEX, n, half, row_chunk,
            self.rows_in_ptr, n, self.spec_ptr, n_cols, 1,
        )
        self.column_forward = _Pass.create(
            COMPLEX_FORWARD, n, n_cols, column_chunk,
            self.spec_ptr, 1, self.spec_ptr, 1, n_cols,
        )
        self.column_backward = _Pass.create(
            COMPLEX_BACKWARD, n, n_cols, column_chunk,
            self.spec_ptr, 1, self.spec_ptr, 1, n_cols,
        )
        self.row_backward = _Pass.create(
            COMPLEX_TO_REAL, n, half, row_chunk,
            self.spec_ptr + half * n_cols, n_cols, self.rows_out_ptr, n, 1,
        )

    cdef void transform_kernel(self, Py_ssize_t num_threads) noexcept nogil:
        self.kernel_rows.run(num_threads)
        self.kernel_columns.run(num_threads)

    cdef void convolve(self, Py_ssize_t num_threads) noexcept nogil:
        """Convolve ``rows_in`` with the kernel, leaving the result in
        ``rows_out`` (unnormalized)."""
        cdef:
            Py_ssize_t n_rows = self.n
            Py_ssize_t half = self.n // 2
            Py_ssize_t n_cols = self.n // 2 + 1
            Py_ssize_t i, j
            double complex *spec = self.spec_ptr
            double complex *kernel_spec = self.kernel_spec_ptr

        self.row_forward.run(num_threads)
        # The column passes overwrote the bottom half of the previous spectrum
        for i in prange(half, n_rows, num_threads=num_threads):
            memset(spec + i * n_cols, 0, n_cols * sizeof(double complex))
        self.column_forward.run(num_threads)
        for i in prange(n_rows, num_threads=num_threads):
            for j in range(n_cols):
                spec[i * n_cols + j] = spec[i * n_cols + j] * kernel_spec[i * n_cols + j]
        self.column_backward.run(num_threads)
        self.row_backward.run(num_threads)


cdef _Plans2D _get_plans_2d(Py_ssize_t n, Py_ssize_t num_threads):
    cdef _Plans2D plans = getattr(_local, "plans_2d", None)
    if plans is None or plans.n != n or plans.num_threads != num_threads:
        # Free the previous size's buffers before allocating the new ones
        _local.plans_2d = None
        plans = _local.plans_2d = _Plans2D(n, num_threads)
    return plans


cdef _Plans1D _get_plans_1d(Py_ssize_t n):
    cdef _Plans1D plans = getattr(_local, "plans_1d", None)
    if plans is None or plans.n != n:
        _local.plans_1d = None
        plans = _local.plans_1d = _Plans1D(n)
    return plans


cdef void matrix_multiply_fft_1d(
    double[::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel vectr K tilde with the w coefficients.

    Parameters
    ----------
    kernel_tilde : memoryview
        The generating vector of the 2d Toeplitz matrix i.e. the kernel
        evaluated all all interpolation points from the left most
        interpolation point, embedded in a circulant matrix (doubled in size
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    num_threads : Py_ssize_t
        Unused; the 1d transforms are too short for threads to pay off.

    """
    cdef:
        Py_ssize_t n_interpolation_points_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t n_spec = n_fft_coeffs // 2 + 1
        Py_ssize_t d, i
        _Plans1D plans = _get_plans_1d(n_fft_coeffs)
        double *kernel_in = _real_data(plans.kernel_in)
        double complex *kernel_spec = _complex_data(plans.kernel_spec)
        double *signal = _real_data(plans.signal)
        double complex *spec = _complex_data(plans.spec)
        double *result = _real_data(plans.result)

    with nogil:
        memcpy(kernel_in, &kernel_tilde[0], n_fft_coeffs * sizeof(double))
        fftw_execute(plans.kernel)

        for d in range(n_terms):
            for i in range(n_interpolation_points_1d):
                signal[i] = w_coefficients[i, d]

            fftw_execute(plans.forward)
            for i in range(n_spec):
                spec[i] = spec[i] * kernel_spec[i]
            fftw_execute(plans.backward)

            # FFTW doesn't normalize the inverse transform
            for i in range(n_interpolation_points_1d):
                out[i, d] = result[n_interpolation_points_1d + i] / n_fft_coeffs


cdef void matrix_multiply_fft_2d(
    double[:, ::1] kernel_tilde,
    double[:, ::1] w_coefficients,
    double[:, ::1] out,
    Py_ssize_t num_threads=1,
):
    """Multiply the the kernel matrix K tilde with the w coefficients.

    Parameters
    ----------
    kernel_tilde : memoryview
        The generating matrix of the 3d Toeplitz tensor i.e. the kernel
        evaluated all all interpolation points from the top left most
        interpolation point, embedded in a circulant matrix (doubled in size
        from (n_interp, n_interp) to (2 * n_interp, 2 * n_interp) and
        symmetrized. See how to embed Toeplitz into circulant matrices.
    w_coefficients : memoryview
        The coefficients calculated in Step 1 of the paper, a
        (n_total_interp, n_terms) matrix. The coefficients are embedded into a
        larger matrix in this function, so no prior embedding is needed.
    out : memoryview
        Output matrix. Must be same size as ``w_coefficients``.
    num_threads : Py_ssize_t
        The most threads to split the transforms across; small grids use
        fewer.

    """
    cdef:
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t half = n_fft_coeffs // 2
        Py_ssize_t d, i, j
        double norm = <double>n_fft_coeffs * n_fft_coeffs
        _Plans2D plans
        double *rows_in
        double *rows_out

    num_threads = max(1, min(num_threads, half // MIN_TRANSFORMS_PER_THREAD))

    plans = _get_plans_2d(n_fft_coeffs, num_threads)
    rows_in = plans.rows_in_ptr
    rows_out = plans.rows_out_ptr

    with nogil:
        memcpy(
            plans.kernel_in_ptr, &kernel_tilde[0, 0],
            n_fft_coeffs * n_fft_coeffs * sizeof(double),
        )
        plans.transform_kernel(num_threads)

        for d in range(n_terms):
            for i in prange(half, num_threads=num_threads):
                for j in range(half):
                    rows_in[i * n_fft_coeffs + j] = w_coefficients[i * half + j, d]

            plans.convolve(num_threads)

            # FFTW doesn't normalize the inverse transform
            for i in prange(half, num_threads=num_threads):
                for j in range(half):
                    out[i * half + j, d] = rows_out[i * n_fft_coeffs + half + j] / norm


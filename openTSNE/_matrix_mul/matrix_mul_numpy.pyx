# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
cimport openTSNE._matrix_mul.matrix_mul
cimport numpy as np
import numpy as np


cdef double[:, ::1] matrix_multiply_fft_1d(
    double[::1] kernel_tilde,
    double[:, ::1] w_coefficients,
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
        
    Returns
    -------
    memoryview
        Contains the kernel values at the equspaced interpolation nodes.
    
    """
    cdef:
        Py_ssize_t n_interpolation_points_1d = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]

        double[:, ::1] y_tilde_values = np.empty((n_interpolation_points_1d, n_terms), dtype=float)

        complex[::1] fft_kernel_tilde = np.empty(n_fft_coeffs, dtype=complex)
        complex[::1] fft_w_coeffs = np.empty(n_fft_coeffs, dtype=complex)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the first half of the vector - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        complex[::1] fft_in_buffer = np.zeros(n_fft_coeffs, dtype=complex)
        complex[::1] fft_out_buffer = np.zeros(n_fft_coeffs, dtype=complex)

        Py_ssize_t d, i

    # Compute the FFT of the kernel vector
    fft_kernel_tilde = np.fft.fft(kernel_tilde)

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            fft_in_buffer[i] = w_coefficients[i, d]

        fft_w_coeffs = np.fft.fft(fft_in_buffer)

        # Take the Hadamard product of two complex vectors
        fft_w_coeffs = np.multiply(fft_w_coeffs, fft_kernel_tilde)

        fft_out_buffer = np.fft.ifft(fft_w_coeffs)

        for i in range(n_interpolation_points_1d):
            y_tilde_values[i, d] = fft_out_buffer[n_interpolation_points_1d + i].real

    return y_tilde_values


cdef double[:, ::1] matrix_multiply_fft_2d(
    double[:, ::1] kernel_tilde,
    double[:, ::1] w_coefficients,
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

    Returns
    -------
    memoryview
        Contains the kernel values at the equspaced interpolation nodes.

    """
    cdef:
        Py_ssize_t total_interpolation_points = w_coefficients.shape[0]
        Py_ssize_t n_terms = w_coefficients.shape[1]
        Py_ssize_t n_fft_coeffs = kernel_tilde.shape[0]
        Py_ssize_t n_interpolation_points_1d = n_fft_coeffs / 2

        double[:, ::1] y_tilde_values = np.empty((total_interpolation_points, n_terms))

        complex[:, :] fft_w_coefficients = np.empty((n_fft_coeffs, (n_fft_coeffs / 2 + 1)), dtype=complex)
        complex[:, :] fft_kernel_tilde = np.empty((n_fft_coeffs, (n_fft_coeffs / 2 + 1)), dtype=complex)
        # Note that we can't use the same buffer for the input and output since
        # we only write to the top quadrant of the in matrix - we'd need to
        # manually zero out the rest of the entries that were inevitably
        # changed during the IDFT, so it's faster to use two buffers, at the
        # cost of some memory
        double[:, ::1] fft_in_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs), dtype=float)
        double[:, ::1] fft_out_buffer = np.zeros((n_fft_coeffs, n_fft_coeffs), dtype=float)

        Py_ssize_t d, i, j, idx

    fft_kernel_tilde = np.fft.rfft2(kernel_tilde)

    for d in range(n_terms):
        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                fft_in_buffer[i, j] = w_coefficients[i * n_interpolation_points_1d + j, d]

        fft_w_coefficients = np.fft.rfft2(fft_in_buffer)

        # Take the Hadamard product of two complex vectors
        fft_w_coefficients = np.multiply(fft_w_coefficients, fft_kernel_tilde)

        # Invert the computed values at the interpolated nodes
        fft_out_buffer = np.fft.irfft2(fft_w_coefficients)

        for i in range(n_interpolation_points_1d):
            for j in range(n_interpolation_points_1d):
                idx = i * n_interpolation_points_1d + j
                y_tilde_values[idx, d] = fft_out_buffer[n_interpolation_points_1d + i,
                                                        n_interpolation_points_1d + j].real

    return y_tilde_values

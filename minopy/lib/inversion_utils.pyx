#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from scipy import linalg as LA
from scipy.linalg import lapack as lap
from libc.math cimport sqrt, log
from scipy.optimize import minimize
from skimage.measure._ccomp import label_cython as clabel


cdef extern from "complex.h":
    double complex cexp(double complex z)
    double complex conj(double complex z)
    double creal(double complex z)
    double cimag(double complex z)
    double cabs(double complex z)
    double complex clog(double complex z)
    double complex csqrt(double complex z)
    double atan2(double y, double x)


cdef get_big_box_cy(double[::1] box, int range_window, int azimuth_window, int width, int length):
    cdef double[4] big_box
    big_box[0] = box[0] - range_window
    big_box[1] = box[1] - azimuth_window
    big_box[2] = box[2] + range_window
    big_box[3] = box[3] + azimuth_window

    if big_box[0] <= 0:
        big_box[0] = 0
    if big_box[1] <= 0:
        big_box[1] = 0
    if big_box[2] > width:
        big_box[2] = width
    if big_box[3] > length:
        big_box[3] = length
    return big_box


cdef double ang(double complex x):
    cdef double r = cabs(x)

    if r == 0:
        r = 1e-100
    cdef double p1 = cimag(x)/r
    cdef double p2 = creal(x)/r

    cdef double out
    out = atan2(p1, p2)

    return out

cdef double[::1] absmat1(double complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef double[::1] y = np.empty((n), dtype='double')

    for i in range(n):
            y[i] = cabs(x[i])
    return y

cdef double[:, ::1] absmat2(double complex[:, ::1] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef double[:, ::1] y = np.empty((n1, n2), dtype='double')

    for i in range(n1):
        for j in range(n2):
            y[i, j] = cabs(x[i, j])
    return y

cdef double[::1] angmat(double complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef double[::1] y = np.empty((n), dtype='double')

    for i in range(n):
            y[i] = ang(x[i])
    return y

cdef double[:, ::1] angmat2(double complex[:, ::1] x):
    cdef cnp.intp_t i, t
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef double[:, ::1] y = np.empty((n1, n2), dtype='double')

    for i in range(n1):
        for t in range(n2):
            y[i, t] = ang(x[i, t])
    return y

cdef double complex[::1] expmati(double[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef double complex[::1] y = np.empty((n), dtype=complex)

    for i in range(n):
            y[i] = cexp(1j * x[i])
    return y


cdef double complex[::1] conjmat1(double complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef double complex[::1] y = np.empty((n), dtype=complex)

    for i in range(n):
            y[i] = conj(x[i])
    return y

cdef double complex[:,::1] conjmat2(double complex[:,::1] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef double complex[:, ::1] y = np.empty((n1, n2), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            y[i, j] = conj(x[i, j])
    return y


cdef double complex[:,::1] multiply_elementwise_dc(double[:, :] x, double complex[:, ::1] y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    cdef cnp.intp_t i, t
    cdef double complex[:, ::1] out = np.zeros((y.shape[0], y.shape[1]), dtype=complex)
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            out[i, t] = y[i,t] * x[i, t]
    return out


cdef double complex multiplymat11(double complex[::1] x, double complex[::1] y):
    assert x.shape[0] == y.shape[0]
    cdef double complex out = 0
    cdef cnp.intp_t i
    for i in range(x.shape[0]):
            out += x[i] * y[i]
    return out


cdef double complex[:,::1] multiplymat22(double complex[:, :] x, double complex[:, ::1] y):
    assert x.shape[1] == y.shape[0]
    cdef cnp.intp_t s1 = x.shape[0]
    cdef cnp.intp_t s2 = y.shape[1]
    cdef double complex[:,::1] out = np.zeros((s1,s2), dtype='complex')
    cdef cnp.intp_t i, t, m

    for i in range(s1):
        for t in range(s2):
            for m in range(x.shape[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out


cdef double complex[::1] multiplymat12(double complex[::1] x, double complex[:, ::1] y):
    assert x.shape[0] == y.shape[0]
    cdef cnp.intp_t s1 = x.shape[0]
    cdef cnp.intp_t s2 = y.shape[1]
    cdef double complex[::1] out = np.zeros((s2), dtype='complex')
    cdef cnp.intp_t i, t, m

    for i in range(s2):
        for t in range(s1):
            out[i] += x[t] * y[t,i]
    return out


cdef bint is_semi_pos_def_chol_cy(double[:, ::1] x):
    """ Checks the positive semi definitness of a matrix. """
    cdef bint res

    try:
        LA.cholesky(x)
        res = True
    except:
        res = False
    return res


cdef tuple regularize_matrix_cy(double[:, ::1] M):
    """ Regularizes a matrix to make it positive semi definite. """
    cdef bint status = False
    cdef cnp.intp_t i, t = 0
    cdef double[:, ::1] N = np.zeros((M.shape[0], M.shape[1]), dtype='double')
    cdef double en = 1e-6

    N[:,:] = M[:,:]

    while t < 500:
        if is_semi_pos_def_chol_cy(N):
            status = True
            break
        else:
            for i in range(M.shape[0]):
                N[i, i] += en
                en *= 2
                t += 1

    return status, N


cdef double complex[::1] EVD_phase_estimation_cy(double complex[:, ::1] coh0):
    """ Estimates the phase values based on eigen value decomosition """
    cdef double[::1] eigen_value
    cdef cnp.ndarray[double complex, ndim=2] eigen_vector
    cdef double complex x0
    cdef cnp.intp_t i, n = coh0.shape[0]
    cdef double complex[::1] vec = np.empty((n), dtype=complex)

    eigen_value, eigen_vector = lap.zheevr(coh0)[0:2]
    x0 = cexp(1j * ang(eigen_vector[0, n-1]))
    for i in range(n):
        vec[i] = eigen_vector[i, n-1] * conj(x0)

    return vec



cdef double complex[::1] EMI_phase_estimation_cy(double complex[:, ::1] coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    cdef double[:,::1] abscoh = absmat2(coh0)
    cdef bint stat
    cdef cnp.intp_t i, n = coh0.shape[0]
    cdef double complex[:, ::1] M
    cdef double[::1] eigen_value
    cdef cnp.ndarray[double complex, ndim=2] eigen_vector
    cdef double complex[::1] vec = np.empty((n), dtype=complex)
    cdef double complex x0

    stat, abscoh = regularize_matrix_cy(abscoh)

    if stat:
        M = lap.zpotri(abscoh)[0] * coh0
        eigen_value, eigen_vector = lap.zheevr(M)[0:2]
        x0 = cexp(1j * ang(eigen_vector[0, 0]))
        for i in range(n):
            vec[i] = eigen_vector[i, 0] * conj(x0)
    else:
        vec = EVD_phase_estimation_cy(coh0)
        printf('warning: coherence matrix not positive semidifinite, It is switched from EMI to EVD')
    return vec


cpdef double optphase_cy(double[::1] x0, double complex[:, ::1] inverse_gam):
    cdef cnp.intp_t n, i
    cdef double complex[::1] x
    cdef double complex[::1] y
    cdef double complex u = 0
    cdef double out = 1

    n = x0.shape[0]
    x = expmati(x0)
    y = multiplymat12(conjmat1(x), inverse_gam)
    u = multiplymat11(y, x)
    out = cabs(clog(u))
    return  out


cdef double[::1] optimize_lbfgs(double[::1] x0, double complex[:, ::1] inverse_gam):
    cdef object res
    cdef double[::1] out = np.zeros((x0.shape[0]), dtype='double')
    cdef cnp.intp_t i

    res = minimize(optphase_cy, x0, args=inverse_gam, method='L-BFGS-B',
                       tol=None, options={'gtol': 1e-6, 'disp': False})

    for i in range(x0.shape[0]):
        out[i] = res.x[i] - res.x[0]

    return out


cdef double complex[::1] PTA_L_BFGS_cy(double complex[:, ::1] coh0):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """
    cdef cnp.intp_t i, n_image = coh0.shape[0]
    cdef double complex[::1] x
    cdef double[::1] x0, amp, res
    cdef double[:, ::1] abscoh = absmat2(coh0)
    cdef double[:, :] invabscoh
    cdef bint stat
    cdef double complex[:, ::1] inverse_gam
    cdef double complex[::1] vec = np.empty((n_image), dtype=complex)

    x = EMI_phase_estimation_cy(coh0)
    x0 = angmat(x)
    amp = absmat1(x)
    stat, abscoh = regularize_matrix_cy(abscoh)
    if stat:
        invabscoh = LA.pinv(abscoh)
        inverse_gam = multiply_elementwise_dc(invabscoh, coh0)
        res = optimize_lbfgs(x0, inverse_gam)
        for i in range(n_image):
            vec[i] = amp[i] * cexp(1j * res[i])

        return vec

    else:

        printf('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation_cy(coh0)


cdef double complex[:,::1] outer_product(double complex[::1] x, double complex[::1] y):
    cdef cnp.intp_t i, t, n = x.shape[0]
    cdef double complex[:, ::1] out = np.empty((n, n), dtype='complex')
    for i in range(n):
        for t in range(n):
            out[i, t] = x[i] * y[t]
    return out


cdef double complex[:,::1] divide_elementwise(double complex[:, ::1] x, double complex[:, ::1] y):
    cdef cnp.intp_t i, t
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef double complex[:, ::1] out = np.empty((n1, n2), dtype='complex')
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == 0:
                out[i, t] = 0
            else:
                out[i, t] = x[i, t] / y[i, t]
    return out

cdef double complex[:, ::1] cov2corr_cy(double complex[:,::1] cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    cdef cnp.intp_t i, n = cov_matrix.shape[0]
    cdef double complex[::1] v = np.empty((n), dtype=complex)
    cdef double complex[:, ::1] outer_v, corr_matrix
    for i in range(n):
        v[i] = csqrt(cov_matrix[i, i])

    outer_v = outer_product(v, v)
    corr_matrix = divide_elementwise(cov_matrix, outer_v)

    return corr_matrix


cdef double complex[:,::1] transposemat2(double complex[:, :] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef double complex[:, ::1] y = np.empty((n2, n1), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            y[j, i] = x[i, j]
    return y


cdef double complex[:,::1] est_corr_cy(double complex[:,::1] CCGS):
    """ Estimate Correlation matrix from an ensemble."""
    cdef cnp.intp_t i, t
    cdef double complex[:,::1] cov_mat, corr_matrix

    cov_mat = multiplymat22(CCGS,  conjmat2(transposemat2(CCGS)))

    for i in range(cov_mat.shape[0]):
        for t in range(cov_mat.shape[1]):
            cov_mat[i, t] /= CCGS.shape[1]

    corr_matrix = cov2corr_cy(cov_mat)

    return corr_matrix


cdef double sum1d(double[::1] x):
    cdef cnp.intp_t i, n = x.shape[0]
    cdef double out = 0
    for i in range(n):
        out += x[i]
    return out


cdef double complex[::1] test_PS_cy(double complex[:, ::1] coh_mat):
    """ checks if the pixel is PS """

    cdef cnp.intp_t i, t, n = coh_mat.shape[0]
    cdef double complex[::1] vec
    cdef double[::1] eigen_value, norm_eigenvalues
    cdef cnp.ndarray[double complex, ndim=2] eigen_vector
    cdef double complex[:, ::1] CM
    cdef int indx = 0
    cdef double s
    cdef bint[:] msk

    Eigen_value, Eigen_vector = lap.zheevr(coh_mat)[0:2]
    norm_eigenvalues = Eigen_value

    s = sum1d(Eigen_value)
    for i in range(n):
        norm_eigenvalues[i] *= (100 / s)
        if norm_eigenvalues[i] > 25:
            indx += 1

    if indx > 0:
        for i in range(n):
            if norm_eigenvalues[i] <= 25:
                Eigen_value[i] = 0

        CM = conjmat2(transposemat2(Eigen_vector))

        for i in range(n):
            for t in range(n):
                CM[i, t] *= sqrt(Eigen_value[i])

        CM = multiplymat22(Eigen_vector, CM)
        vec = EMI_phase_estimation_cy(CM)
    else:
        vec = EMI_phase_estimation_cy(coh_mat)

    return vec


cdef double norm_complex(double complex[::1] x):
    cdef cnp.intp_t n = x.shape[0]
    cdef int i
    cdef double out = 0
    for i in range(n):
        out += cabs(x[i])**2
    out = sqrt(out)
    return out

cdef double complex[::1] squeeze_images(double complex[::1] x, double complex[:, ::1] ccg, cnp.intp_t step):
    cdef cnp.intp_t n = x.shape[0]
    cdef cnp.intp_t s = ccg.shape[1]
    cdef double complex[::1] vm = np.zeros((n-step), dtype=complex)
    cdef int i, t
    cdef double normm = 0
    cdef double complex[::1] out = np.zeros((s), dtype=complex)

    for i in range(n - step):
        vm[i] = x[i + step]
    normm = norm_complex(vm)
    for t in range(s):
        for i in range(vm.shape[0]):
            vm[i] /= normm
            out[t] += ccg[i + step, t] * conj(vm[i])

    return out

cdef tuple phase_linking_process_cy(double complex[:, ::1] ccg_sample, cnp.int_t stepp,
                                     str method, bint squeez=True):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """

    cdef double complex[:, ::1] coh_mat
    cdef double complex[::1] res
    cdef cnp.intp_t n1 = ccg_sample.shape[1]
    cdef double complex[::1] squeezed = np.empty((n1), dtype=complex)

    coh_mat = est_corr_cy(ccg_sample)

    if method == 'PTA' or method == 'sequential_PTA':
        res = PTA_L_BFGS_cy(coh_mat)
    elif method == 'EMI' or method == 'sequential_EMI':
        res = EMI_phase_estimation_cy(coh_mat)
    else:
        res = EVD_phase_estimation_cy(coh_mat)

    if squeez:
        squeezed = squeeze_images(res, ccg_sample, stepp)


    return res, squeezed



cdef tuple sequential_phase_linking_cy(double complex[:,::1] full_stack_complex_samples,
                                        str method, int mini_stack_default_size,
                                        int total_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    cdef int i, t, sstep, first_line, last_line, num_lines
    cdef int a1, a2
    cdef cnp.intp_t n_image = full_stack_complex_samples.shape[0]
    cdef double complex[::1] vec_refined = np.zeros((n_image), dtype=complex)
    cdef double complex[:, ::1] mini_stack_complex_samples
    cdef double complex[::1] res, squeezed_images_0
    cdef double complex[:, ::1] squeezed_images = np.zeros((total_num_mini_stacks,
                                                                full_stack_complex_samples.shape[1]),
                                                               dtype=complex)

    for sstep in range(total_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == total_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images_0 = phase_linking_process_cy(mini_stack_complex_samples, sstep, method)

        else:

            mini_stack_complex_samples = np.zeros((sstep + num_lines,
                                                   full_stack_complex_samples.shape[1]), dtype=complex)
            for i in range(full_stack_complex_samples.shape[1]):
                for t in range(sstep):
                    mini_stack_complex_samples[t, i] = squeezed_images[t, i]
                for t in range(num_lines):
                    mini_stack_complex_samples[t + sstep, i] = full_stack_complex_samples[first_line + t, i]

            res, squeezed_images_0 = phase_linking_process_cy(mini_stack_complex_samples, sstep, method)

        for i in range(num_lines):
            vec_refined[first_line + i] = res[sstep + i]

        for i in range(squeezed_images_0.shape[0]):
                squeezed_images[sstep, i] = squeezed_images_0[i]


    return vec_refined, squeezed_images


cdef double complex[::1] datum_connect_cy(double complex[:, ::1] squeezed_images,
                                        double complex[::1] vector_refined, int mini_stack_size):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector

    Returns
    -------

    """
    cdef double[::1] datum_shift
    cdef double complex[::1] new_vector_refined
    cdef int step, i, first_line, last_line

    datum_shift = angmat(phase_linking_process_cy(squeezed_images, 0, 'PTA', squeez=False)[0])
    new_vector_refined = np.zeros((vector_refined.shape[0]), dtype=complex)

    for step in range(datum_shift.shape[0]):
        first_line = step * mini_stack_size
        if step == datum_shift.shape[0] - 1:
            last_line = vector_refined.shape[0]
        else:
            last_line = first_line + mini_stack_size

        for i in range(last_line - first_line):
            new_vector_refined[i + first_line] = vector_refined[i + first_line] * cexp(1j * datum_shift[step])

    return new_vector_refined


cdef double searchsorted_max(cnp.ndarray[double, ndim=1] x1, cnp.ndarray[double, ndim=1] x2,
                               cnp.ndarray[double, ndim=1] y):
    cdef int nx = x1.shape[0]
    cdef int ny = y.shape[0]
    cdef double outtmp, out = 0
    cdef int t1, t2, i = 0
    cdef int temp1 = 0
    cdef int temp2 = 0
    while i < ny:
        t1 = temp1
        t2 = temp2
        while t1 <= nx:
            if y[i] >= x1[t1]:
                temp1 = t1
                break
            t1 += 1

        while t2 <= nx:
            if y[i] >= x2[t2]:
                temp2 = t2
                break
            t2 += 1

        outtmp = abs(temp1 - temp2)/nx
        if outtmp > out:
            out = outtmp
        i += 1
    return out


cdef void sorting(cnp.ndarray[double, ndim=1] x):
    x.sort()
    return


cdef float ecdf_distance(cnp.ndarray[double, ndim=1] data1, cnp.ndarray[double, ndim=1] data2):

    cdef int n1 = data1.shape[0]
    cdef cnp.ndarray[double, ndim=1] data_all = concat_cy(data1, data2)
    cdef float distance

    sorting(data_all)
    distance = searchsorted_max(data1, data2, data_all)

    return distance



cdef cnp.ndarray[double, ndim=1] concat_cy(cnp.ndarray[double, ndim=1] x, cnp.ndarray[double, ndim=1] y):
    cdef int n1 = x.shape[0]
    cdef int n2 = y.shape[0]
    cdef cnp.ndarray[double, ndim=1] out = np.zeros((n1 + n2), dtype='double')
    cdef int i
    for i in range(n1):
        out[i] = x[i]
        out[i + n1] = y[i]
    return out



cdef int count(cnp.ndarray[long, ndim=2]  x, long value,):
    cdef int n1 = x.shape[0]
    cdef int n2 = x.shape[1]
    cdef int i, t, out = 0
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == value:
                out += 1
    return out



cdef int[:, ::1] get_shp_row_col_c((int, int) data, double complex[:, :, ::1] input_slc,
                        int[::1] def_sample_rows, int[::1] def_sample_cols,
                        int azimuth_window, int range_window, int reference_row,
                        int reference_col, float distance_threshold):

    cdef int row_0, col_0, i, temp, ref_row, ref_col, t1, t2, s_rows, s_cols
    cdef long ref_label
    cdef cnp.intp_t width, length, n_image = input_slc.shape[0]
    cdef int[::1] sample_rows, sample_cols
    cdef cnp.ndarray[long, ndim=2] ks_label, distance
    cdef int[:, ::1] shps
    cdef cnp.ndarray[double, ndim=1] S1 = np.zeros((n_image), dtype='double')
    cdef cnp.ndarray[double, ndim=1] test = np.zeros((n_image), dtype='double')

    row_0 = data[0]
    col_0 = data[1]
    length = input_slc.shape[1]
    width = input_slc.shape[2]


    t1 = 0
    t2 = def_sample_rows.shape[0]
    for i in range(def_sample_rows.shape[0]):
        temp = row_0 + def_sample_rows[i]
        if temp < 0:
            t1 += 1
        if temp >= length and t2 == def_sample_rows.shape[0]:
            t2 = i + 1

    s_rows = t2 - t1
    ref_row = reference_row - t1

    sample_rows = np.zeros((s_rows), dtype=np.int32)
    for i in range(s_rows):
        sample_rows[i] = row_0 + def_sample_rows[i + t1]

    t1 = 0
    t2 = def_sample_cols.shape[0]
    for i in range(def_sample_cols.shape[0]):
        temp = col_0 + def_sample_cols[i]
        if temp < 0:
            t1 += 1
        if temp >= width and t2 == def_sample_cols.shape[0]:
            t2 = i + 1

    s_cols = t2 - t1
    ref_col = reference_col - t1

    sample_cols = np.zeros((s_cols), dtype=np.int32)
    for i in range(s_cols):
        sample_cols[i] = col_0 + def_sample_cols[i + t1]


    for i in range(n_image):
        S1[i] = cabs(input_slc[i, row_0, col_0])

    sorting(S1)

    distance = np.zeros((s_rows, s_cols), dtype=long)

    for t1 in range(s_rows):
        for t2 in range(s_cols):
            for temp in range(n_image):
                test[temp] = cabs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
            sorting(test)
            distance[t1, t2] = 1 * (ecdf_distance(S1, test) <= distance_threshold)

    ks_label = clabel(distance, connectivity=2)
    ref_label = ks_label[ref_row, ref_col]

    temp = count(ks_label, ref_label)
    shps = np.zeros((temp, 2), dtype=np.int32)

    temp = 0
    for t1 in range(s_rows):
        for t2 in range(s_cols):
            if ks_label[t1, t2] == ref_label:

                shps[temp, 0] = sample_rows[t1]
                shps[temp, 1] = sample_cols[t2]
                temp += 1

    return shps




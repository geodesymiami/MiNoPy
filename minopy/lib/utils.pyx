#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from scipy import linalg as LA
from scipy.linalg import lapack as lap
from libc.math cimport sqrt, log, exp
from scipy.optimize import minimize
from skimage.measure._ccomp import label_cython as clabel
from scipy.stats import anderson_ksamp, ttest_ind
from mintpy.utils import ptime
#from libcpp.string cimport string


cdef extern from "complex.h":
    float complex cexpf(float complex z)
    float complex conjf(float complex z)
    float crealf(float complex z)
    float cimagf(float complex z)
    float cabsf(float complex z)
    float complex clogf(float complex z)
    float complex csqrtf(float complex z)
    float cargf(float complex z)

cdef float cargf_r(float complex z):
    cdef float res
    res = cargf(z)
    #if res == nan:
    #    res = 0
    return res

cdef cnp.ndarray[int, ndim=1] get_big_box_cy(cnp.ndarray[int, ndim=1] box, int range_window, int azimuth_window, int width, int length):
    cdef cnp.ndarray[int, ndim=1] big_box = np.arange(4, dtype=np.int32)
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


cdef float[::1] absmat1(float complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef float[::1] y = np.empty((n), dtype=np.float32)

    for i in range(n):
            y[i] = cabsf(x[i])
    return y

cdef float[:, ::1] absmat2(float complex[:, ::1] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef float[:, ::1] y = np.empty((n1, n2), dtype=np.float32)

    for i in range(n1):
        for j in range(n2):
            y[i, j] = cabsf(x[i, j])
    return y

cdef float[::1] angmat(float complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef float[::1] y = np.empty((n), dtype=np.float32)

    for i in range(n):
            y[i] = cargf_r(x[i])
    return y

cdef float[:, ::1] angmat2(float complex[:, ::1] x):
    cdef cnp.intp_t i, t
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef float[:, ::1] y = np.empty((n1, n2), dtype=np.float32)

    for i in range(n1):
        for t in range(n2):
            y[i, t] = cargf_r(x[i, t])
    return y

cdef float complex[::1] expmati(float[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef float complex[::1] y = np.empty(n, dtype=np.complex64)

    for i in range(n):
            y[i] = cexpf(1j * x[i])
    return y


cdef float complex[::1] conjmat1(float complex[::1] x):
    cdef cnp.intp_t i
    cdef cnp.intp_t n = x.shape[0]
    cdef float complex[::1] y = np.empty(n, dtype=np.complex64)

    for i in range(n):
            y[i] = conjf(x[i])
    return y

cdef float complex[:,::1] conjmat2(float complex[:,::1] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef float complex[:, ::1] y = np.empty((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for j in range(n2):
            y[i, j] = conjf(x[i, j])
    return y


cdef float complex[:,::1] multiply_elementwise_dc(float[:, :] x, float complex[:, ::1] y):
    #assert x.shape[0] == y.shape[0]
    #assert x.shape[1] == y.shape[1]
    cdef cnp.intp_t i, t
    cdef float complex[:, ::1] out = np.zeros((y.shape[0], y.shape[1]), dtype=np.complex64)
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            out[i, t] = y[i,t] * x[i, t]
    return out


cdef float complex multiplymat11(float complex[::1] x, float complex[::1] y):
    #assert x.shape[0] == y.shape[0]
    cdef float complex out = 0
    cdef cnp.intp_t i
    for i in range(x.shape[0]):
            out += x[i] * y[i]
    return out


cdef float complex[:,::1] multiplymat22(float complex[:, :] x, float complex[:, ::1] y):
    #assert x.shape[1] == y.shape[0]
    cdef cnp.intp_t s1 = x.shape[0]
    cdef cnp.intp_t s2 = y.shape[1]
    cdef float complex[:,::1] out = np.zeros((s1,s2), dtype=np.complex64)
    cdef cnp.intp_t i, t, m

    for i in range(s1):
        for t in range(s2):
            for m in range(x.shape[1]):
                out[i,t] += x[i,m] * y[m,t]

    return out


cdef float complex[::1] multiplymat12(float complex[::1] x, float complex[:, ::1] y):
    #assert x.shape[0] == y.shape[0]
    cdef cnp.intp_t s1 = x.shape[0]
    cdef cnp.intp_t s2 = y.shape[1]
    cdef float complex[::1] out = np.zeros(s2, dtype=np.complex64)
    cdef cnp.intp_t i, t, m

    for i in range(s2):
        for t in range(s1):
            out[i] += x[t] * y[t,i]
    return out


cdef bint is_semi_pos_def_chol_cy(float[:, ::1] x):
    """ Checks the positive semi definitness of a matrix. """
    cdef bint res

    try:
        LA.cholesky(x)
        res = True
    except:
        res = False
    return res


cdef tuple regularize_matrix_cy(float[:, ::1] M):
    """ Regularizes a matrix to make it positive semi definite. """
    cdef bint status = False
    cdef cnp.intp_t i, t = 0
    cdef float[:, ::1] N = np.zeros((M.shape[0], M.shape[1]), dtype=np.float32)
    cdef float en = 1e-6

    for i in range(M.shape[0]):
        for t in range(M.shape[1]):
            N[i, t] = M[i, t]

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


cdef float complex[::1] EVD_phase_estimation_cy(float complex[:, ::1] coh0):
    """ Estimates the phase values based on eigen value decomosition """
    cdef float[::1] eigen_value
    cdef cnp.ndarray[float complex, ndim=2] eigen_vector
    cdef float complex x0
    cdef cnp.intp_t i, n = coh0.shape[0]
    cdef float complex[::1] vec = np.empty(n, dtype=np.complex64)

    eigen_value, eigen_vector = lap.cheevr(coh0)[0:2]
    x0 = cexpf(1j * cargf_r(eigen_vector[0, n-1]))
    for i in range(n):
        vec[i] = eigen_vector[i, n-1] * conjf(x0)

    return vec



cdef float complex[::1] EMI_phase_estimation_cy(float complex[:, ::1] coh0):
    """ Estimates the phase values based on EMI decomosition (Homa Ansari, 2018 paper) """
    cdef float[:,::1] abscoh = absmat2(coh0)
    cdef bint stat
    cdef cnp.intp_t i, n = coh0.shape[0]
    cdef float complex[:, ::1] M
    cdef float[::1] eigen_value
    cdef cnp.ndarray[float complex, ndim=2] eigen_vector
    cdef float complex[::1] vec = np.empty((n), dtype=np.complex64)
    cdef float complex x0

    stat, abscoh = regularize_matrix_cy(abscoh)

    if stat:
        M = lap.cpotri(abscoh)[0] * coh0
        eigen_value, eigen_vector = lap.cheevr(M)[0:2]
        x0 = cexpf(1j * cargf_r(eigen_vector[0, 0]))
        for i in range(n):
            vec[i] = eigen_vector[i, 0] * conjf(x0)
    else:
        vec = EVD_phase_estimation_cy(coh0)
        printf('warning: coherence matrix not positive semidifinite, It is switched from EMI to EVD')
    return vec


cpdef double optphase_cy(double[::1] x0, float complex[:, ::1] inverse_gam):
    cdef cnp.intp_t n, i
    cdef float complex[::1] x
    cdef float complex[::1] y
    cdef float complex u = 0
    cdef double out = 1

    n = x0.shape[0]
    x = expmati(np.float32(x0))
    y = multiplymat12(conjmat1(x), inverse_gam)
    u = multiplymat11(y, x)
    out = cabsf(clogf(u))
    return  out

cdef float[::1] optimize_lbfgs(float[::1] x0, float complex[:, ::1] inverse_gam):
    cdef double[::1] res
    cdef float[::1] out = np.zeros(x0.shape[0], dtype=np.float32)
    cdef cnp.intp_t i

    res = minimize(optphase_cy, x0, args=inverse_gam, method='L-BFGS-B', tol=None, options={'gtol': 1e-6, 'disp': False}).x

    for i in range(x0.shape[0]):
        out[i] = res[i] - res[0]

    return out


cdef float complex[::1] PTA_L_BFGS_cy(float complex[:, ::1] coh0):
    """ Uses L-BFGS method to optimize PTA function and estimate phase values. """
    cdef cnp.intp_t i, n_image = coh0.shape[0]
    cdef float complex[::1] x
    cdef float[::1] x0, amp, res
    cdef float[:, ::1] abscoh = absmat2(coh0)
    cdef float[:, :] invabscoh
    cdef bint stat
    cdef float complex[:, ::1] inverse_gam
    cdef float complex[::1] vec = np.empty(n_image, dtype=np.complex64)

    x = EMI_phase_estimation_cy(coh0)
    x0 = angmat(x)
    amp = absmat1(x)
    stat, abscoh = regularize_matrix_cy(abscoh)
    if stat:
        invabscoh = lap.spotri(abscoh)[0]
        inverse_gam = multiply_elementwise_dc(invabscoh, coh0)
        res = optimize_lbfgs(x0, inverse_gam)
        for i in range(n_image):
            vec[i] = amp[i] * cexpf(1j * res[i])

        return vec

    else:

        printf('warning: coherence matrix not positive semidifinite, It is switched from PTA to EVD')
        return EVD_phase_estimation_cy(coh0)


cdef float complex[:,::1] outer_product(float complex[::1] x, float complex[::1] y):
    cdef cnp.intp_t i, t, n = x.shape[0]
    cdef float complex[:, ::1] out = np.empty((n, n), dtype=np.complex64)
    for i in range(n):
        for t in range(n):
            out[i, t] = x[i] * y[t]
    return out


cdef float complex[:,::1] divide_elementwise(float complex[:, ::1] x, float complex[:, ::1] y):
    cdef cnp.intp_t i, t
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef float complex[:, ::1] out = np.empty((n1, n2), dtype=np.complex64)
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == 0:
                out[i, t] = 0
            else:
                out[i, t] = x[i, t] / y[i, t]
    return out

cdef float complex[:, ::1] cov2corr_cy(float complex[:,::1] cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """
    cdef cnp.intp_t i, n = cov_matrix.shape[0]
    cdef float complex[::1] v = np.empty(n, dtype=np.complex64)
    cdef float complex[:, ::1] outer_v, corr_matrix
    for i in range(n):
        v[i] = csqrtf(cov_matrix[i, i])

    outer_v = outer_product(v, v)
    corr_matrix = divide_elementwise(cov_matrix, outer_v)

    return corr_matrix


cdef float complex[:,::1] transposemat2(float complex[:, :] x):
    cdef cnp.intp_t i, j
    cdef cnp.intp_t n1 = x.shape[0]
    cdef cnp.intp_t n2 = x.shape[1]
    cdef float complex[:, ::1] y = np.empty((n2, n1), dtype=np.complex64)
    for i in range(n1):
        for j in range(n2):
            y[j, i] = x[i, j]
    return y


cdef float complex[:,::1] est_corr_cy(float complex[:,::1] ccg):
    """ Estimate Correlation matrix from an ensemble."""
    cdef cnp.intp_t i, t
    cdef float complex[:,::1] cov_mat, corr_matrix

    cov_mat = multiplymat22(ccg,  conjmat2(transposemat2(ccg)))

    for i in range(cov_mat.shape[0]):
        for t in range(cov_mat.shape[1]):
            cov_mat[i, t] /= ccg.shape[1]

    corr_matrix = cov2corr_cy(cov_mat)

    return corr_matrix


cdef float sum1d(float[::1] x):
    cdef cnp.intp_t i, n = x.shape[0]
    cdef float out = 0
    for i in range(n):
        out += x[i]
    return out


cdef float complex[::1] test_PS_cy(float complex[:, ::1] coh_mat):
    """ checks if the pixel is PS """

    cdef cnp.intp_t i, t, n = coh_mat.shape[0]
    cdef float complex[::1] vec
    cdef float[::1] eigen_value, norm_eigenvalues
    cdef cnp.ndarray[float complex, ndim=2] eigen_vector
    cdef float complex[:, ::1] CM
    cdef int indx = 0
    cdef float s
    cdef bint[:] msk

    Eigen_value, Eigen_vector = lap.cheevr(coh_mat)[0:2]
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


cdef float norm_complex(float complex[::1] x):
    cdef cnp.intp_t n = x.shape[0]
    cdef int i
    cdef float out = 0
    for i in range(n):
        out += cabsf(x[i])**2
    out = sqrt(out)
    return out

cdef float complex[::1] squeeze_images(float complex[::1] x, float complex[:, ::1] ccg, cnp.intp_t step):
    cdef cnp.intp_t n = x.shape[0]
    cdef cnp.intp_t s = ccg.shape[1]
    cdef float complex[::1] vm = np.zeros((n-step), dtype=np.complex64)
    cdef int i, t
    cdef float normm = 0
    cdef float complex[::1] out = np.zeros((s), dtype=np.complex64)

    for i in range(n - step):
        vm[i] = x[i + step]
    normm = norm_complex(vm)
    for t in range(s):
        for i in range(vm.shape[0]):
            vm[i] /= normm
            out[t] += ccg[i + step, t] * conjf(vm[i])

    return out

cdef tuple phase_linking_process_cy(float complex[:, ::1] ccg_sample, int stepp, bytes method, bint squeez):
    """Inversion of phase based on a selected method among PTA, EVD and EMI """

    cdef float complex[:, ::1] coh_mat
    cdef float complex[::1] res
    cdef cnp.intp_t n1 = ccg_sample.shape[1]
    cdef float complex[::1] squeezed #= np.empty(n1, dtype=np.complex64)

    coh_mat = est_corr_cy(ccg_sample)

    if method.decode('utf-8') == 'PTA' or method.decode('utf-8') == 'sequential_PTA':
        res = PTA_L_BFGS_cy(coh_mat)
    elif method.decode('utf-8') == 'EMI' or method.decode('utf-8') == 'sequential_EMI':
        res = EMI_phase_estimation_cy(coh_mat)
    else:
        res = EVD_phase_estimation_cy(coh_mat)

    if squeez:
        squeezed = squeeze_images(res, ccg_sample, stepp)
        return res, squeezed
    else:
        return res, 0


cdef tuple sequential_phase_linking_cy(float complex[:,::1] full_stack_complex_samples,
                                        bytes method, int mini_stack_default_size,
                                        int total_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    cdef int i, t, sstep, first_line, last_line, num_lines
    cdef int a1, a2
    cdef cnp.intp_t n_image = full_stack_complex_samples.shape[0]
    cdef float complex[::1] vec_refined = np.zeros((n_image), dtype=np.complex64)
    cdef float complex[:, ::1] mini_stack_complex_samples
    cdef float complex[::1] res, squeezed_images_0
    cdef float complex[:, ::1] squeezed_images = np.zeros((total_num_mini_stacks,
                                                                full_stack_complex_samples.shape[1]),
                                                               dtype=np.complex64)

    for sstep in range(total_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == total_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images_0 = phase_linking_process_cy(mini_stack_complex_samples, sstep, method, True)

        else:

            mini_stack_complex_samples = np.zeros((sstep + num_lines,
                                                   full_stack_complex_samples.shape[1]), dtype=np.complex64)
            for i in range(full_stack_complex_samples.shape[1]):
                for t in range(sstep):
                    mini_stack_complex_samples[t, i] = squeezed_images[t, i]
                for t in range(num_lines):
                    mini_stack_complex_samples[t + sstep, i] = full_stack_complex_samples[first_line + t, i]

            res, squeezed_images_0 = phase_linking_process_cy(mini_stack_complex_samples, sstep, method, True)

        for i in range(num_lines):
            vec_refined[first_line + i] = res[sstep + i]

        for i in range(squeezed_images_0.shape[0]):
                squeezed_images[sstep, i] = squeezed_images_0[i]


    return vec_refined, squeezed_images


cdef float complex[::1] datum_connect_cy(float complex[:, ::1] squeezed_images,
                                        float complex[::1] vector_refined, int mini_stack_size):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector

    Returns
    -------

    """
    cdef float[::1] datum_shift
    cdef float complex[::1] new_vector_refined
    cdef int step, i, first_line, last_line
    cdef bytes method = b'EMI'

    datum_shift = angmat(phase_linking_process_cy(squeezed_images, 0, method, False)[0])
    new_vector_refined = np.zeros((vector_refined.shape[0]), dtype=np.complex64)

    for step in range(datum_shift.shape[0]):
        first_line = step * mini_stack_size
        if step == datum_shift.shape[0] - 1:
            last_line = vector_refined.shape[0]
        else:
            last_line = first_line + mini_stack_size

        for i in range(last_line - first_line):
            new_vector_refined[i + first_line] = vector_refined[i + first_line] * cexpf(1j * datum_shift[step])

    return new_vector_refined

@cython.cdivision(False)
cdef float searchsorted_max(cnp.ndarray[float, ndim=1] x1, cnp.ndarray[float, ndim=1] x2, cnp.ndarray[float, ndim=1] y):
    cdef int nx = x1.shape[0]
    cdef int ny = y.shape[0]
    cdef float outtmp, out = 0
    cdef int t1, t2, i = 0
    cdef int temp1 = 0
    cdef int temp2 = 0

    for i in range(ny):

        t1 = temp1
        t2 = temp2

        if y[i] >= x1[nx - 1]:
            temp1 = nx

        if y[i] >= x2[nx - 1]:
            temp2 = nx

        while t1 < nx:
            if t1 == 0 and y[i] < x1[t1]:
                temp1 = 0
                t1 = nx
            elif x1[t1 - 1] <= y[i] and y[i] < x1[t1]:
                temp1 = t1
                t1 = nx
            else:
                t1 += 1

        while t2 < nx:
            if t2 == 0 and y[i] < x2[t2]:
                temp2 = 0
                t2 = nx
            elif x2[t2 - 1] <= y[i] and y[i] < x2[t2]:
                temp2 = t2
                t2 = nx
            else:
                t2 += 1

        outtmp = abs(temp1 - temp2) / nx
        if outtmp > out:
            out = outtmp
    return out


cdef void sorting(cnp.ndarray[float, ndim=1] x):
    x.sort()
    return


cdef float ecdf_distance(cnp.ndarray[float, ndim=1] data1, cnp.ndarray[float, ndim=1] data2):

    cdef int n1 = data1.shape[0]
    cdef cnp.ndarray[float, ndim=1] data_all = concat_cy(data1, data2)
    cdef float distance

    sorting(data_all)
    distance = searchsorted_max(data1, data2, data_all)

    return distance


cdef float ks_lut_cy(int N1, int N2, float alpha):
    cdef float N = (N1 * N2) / (N1 + N2)
    cdef float[::1] distances = np.arange(0.01, 1, 0.001, dtype=np.float32)
    cdef float value, pvalue
    cdef int i, t
    for i in range(distances.shape[0]):
        value = distances[i]*(sqrt(N) + 0.12 + 0.11/sqrt(N))
        pvalue = 0
        for t in range(1, 101):
            pvalue += ((-1)**(t-1))*exp(-2*(value**2)*(t**2))
        pvalue = 2 * pvalue
        if pvalue > 1:
            pvalue = 1
        if pvalue < 0:
            pvalue = 0
        if pvalue <= alpha:
            critical_distance = distances[i]
            break
    return critical_distance


cdef cnp.ndarray[float, ndim=1] concat_cy(cnp.ndarray[float, ndim=1] x, cnp.ndarray[float, ndim=1] y):
    cdef int n1 = x.shape[0]
    cdef int n2 = y.shape[0]
    cdef cnp.ndarray[float, ndim=1] out = np.zeros((n1 + n2), dtype=np.float32)
    cdef int i
    for i in range(n1):
        out[i] = x[i]
        out[i + n1] = y[i]
    return out



cdef int count(cnp.ndarray[long, ndim=2]  x, long value):
    cdef int n1 = x.shape[0]
    cdef int n2 = x.shape[1]
    cdef int i, t, out = 0
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == value:
                out += 1
    return out



cdef int[:, ::1] get_shp_row_col_c((int, int) data, float complex[:, :, ::1] input_slc,
                        int[::1] def_sample_rows, int[::1] def_sample_cols,
                        int azimuth_window, int range_window, int reference_row,
                        int reference_col, float distance_threshold):

    cdef int row_0, col_0, i, temp, ref_row, ref_col, t1, t2, s_rows, s_cols
    cdef long ref_label
    cdef cnp.intp_t width, length, n_image = input_slc.shape[0]
    cdef int[::1] sample_rows, sample_cols
    cdef cnp.ndarray[long, ndim=2] ks_label, distance
    cdef int[:, ::1] shps
    cdef cnp.ndarray[float, ndim=1] ref = np.zeros((n_image), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] test = np.zeros((n_image), dtype=np.float32)

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
        ref[i] = cabsf(input_slc[i, row_0, col_0])

    sorting(ref)

    distance = np.zeros((s_rows, s_cols), dtype=long)

    for t1 in range(s_rows):
        for t2 in range(s_cols):
            for temp in range(n_image):
                test[temp] = cabsf(input_slc[temp, sample_rows[t1], sample_cols[t2]])
            sorting(test)
            distance[t1, t2] = 1 * (ecdf_distance(ref, test) <= distance_threshold)

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

cdef float[::1] mean_along_axis_x(float[:, ::1] x):
    cdef int i, t, n = x.shape[1]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef float temp = 0
    for i in range(n):
        temp = 0
        for t in range(x.shape[0]):
            temp += x[t, i]
        out[i] = temp/x.shape[0]
    return out

cdef float gam_pta_c(float[:, ::1] ph_filt, float complex[::1] vec):
    """ Returns squeesar PTA coherence between the initial and estimated phase vectors.
    :param ph_filt: np.angle(coh) before inversion
    :param vec_refined: refined complex vector after inversion
    """

    cdef int i, k, n = vec.shape[0]
    cdef float[::1] ang_vec = angmat(vec)
    cdef float temp_coh = 0
    cdef float complex temp = 0
    for i in range(n):
        for k in range(i + 1, n):
            temp += cexpf(1j * (ph_filt[i,k] - (ang_vec[i] - ang_vec[k])))

    temp_coh = crealf(temp) * 2 /(n**2 - n)

    return temp_coh


def process_patch_c(cnp.ndarray[int, ndim=1] box, int range_window, int azimuth_window, int width, int length, int n_image,
                      object slcStackObj, float distance_threshold, int[::1] def_sample_rows,
                      int[::1] def_sample_cols, int reference_row, int reference_col, bytes phase_linking_method,
                      int total_num_mini_stacks, int default_mini_stack_size):
    cdef cnp.ndarray[int, ndim=1] big_box = get_big_box_cy(box, range_window, azimuth_window, width, length)
    cdef int box_width = box[2] - box[0]
    cdef int box_length = box[3] - box[1]
    cdef cnp.ndarray[float complex, ndim=3] rslc_ref = np.zeros((n_image, box_length, box_width), dtype=np.complex64)
    cdef cnp.ndarray[float, ndim=2] quality = np.zeros((box_length, box_width), dtype=np.float32)
    cdef int row1 = box[1] - big_box[1]
    cdef int row2 = box[3] - big_box[1]
    cdef int col1 = box[0] - big_box[0]
    cdef int col2 = box[2] - big_box[0]
    cdef int[::1] lin = np.arange(row1, row2, dtype=np.int32)
    cdef int overlap_length = row2 - row1
    cdef int[::1] sam = np.arange(col1, col2, dtype=np.int32)
    cdef int overlap_width = col2 - col1
    cdef int[:, ::1] coords = np.zeros((overlap_length*overlap_width, 2), dtype=np.int32)
    cdef int noval, num_points, num_shp, i, t, p, m = 0
    cdef (int, int) data
    cdef int[:, ::1] shp
    cdef cnp.ndarray[float complex, ndim=3] patch_slc_images = slcStackObj.read(datasetName='slc', box=big_box)
    cdef float complex[:, ::1] CCG, coh_mat, squeezed_images
    cdef float complex[::1] vec_refined, squeezed_images_0
    cdef float[::1] amp_refined
    cdef object prog_bar

    for i in range(overlap_length):
        for t in range(overlap_width):
            coords[m, 0] = i + row1
            coords[m, 1] = t + col1
            m += 1

    num_points = m
    prog_bar = ptime.progressBar(maxValue=num_points)
    p = 0
    for i in range(num_points):
        data = (coords[i,0], coords[i,1])

        shp = get_shp_row_col_c(data, patch_slc_images, def_sample_rows, def_sample_cols, azimuth_window,
                                range_window, reference_row, reference_col, distance_threshold)

        num_shp = shp.shape[0]

        CCG = np.zeros((n_image, num_shp), dtype=np.complex64)
        for t in range(num_shp):
            for m in range(n_image):
                CCG[m, t] = patch_slc_images[m, shp[t,0], shp[t,1]]

        coh_mat = est_corr_cy(CCG)

        if num_shp > 20:

            if len(phase_linking_method) > 10 and phase_linking_method[0:10] == b'sequential':
                vec_refined, squeezed_images = sequential_phase_linking_cy(CCG, phase_linking_method,
                                                                           default_mini_stack_size,
                                                                           total_num_mini_stacks)

                vec_refined = datum_connect_cy(squeezed_images, vec_refined, default_mini_stack_size)

            else:
                vec_refined, noval = phase_linking_process_cy(CCG, 0, phase_linking_method, False)

        else:
            vec_refined = test_PS_cy(coh_mat)

        amp_refined = mean_along_axis_x(absmat2(CCG))

        for m in range(n_image):

            if cabsf(vec_refined[m]) == 0:
                vec_refined[m] = 0
            else:
                vec_refined[m] = (vec_refined[m] / cabsf(vec_refined[m])) * amp_refined[m]
            rslc_ref[m, data[0] - row1, data[1] - col1] = vec_refined[m]

        quality[data[0] - row1, data[1] - col1] = gam_pta_c(angmat2(coh_mat), vec_refined)

        prog_bar.update(p+ 1, every=10, suffix='{}/{} pixels'.format(p + 1, num_points))
        p += 1

    return rslc_ref, quality, box

cdef bint ks2smapletest_cy(float[::1] S1, float[::1] S2, float threshold):
    cdef bint res
    cdef float distance = ecdf_distance(S1, S2)
    if distance <= threshold:
        res = 1
    else:
        res = 0
    return res

cdef bint ttest_indtest_cy(float[::1] S1, float[::1] S2, float threshold):
    cdef object testobj = ttest_ind(S1, S2, equal_var=False)
    cdef float test = testobj[1]
    cdef bint res
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res


cdef bint ADtest_cy(float[::1] S1, float[::1] S2, float threshold):
    cdef object testobj = anderson_ksamp([S1, S2])
    cdef float test = testobj.significance_level
    cdef bint res
    if test >= threshold:
        res = 1
    else:
        res = 0

    return res


cimport numpy as cnp
cimport cython

ctypedef float float

cdef bint isnanc(float complex)
cdef cnp.ndarray[int, ndim=1] get_big_box_cy(cnp.ndarray[int, ndim=1], int, int, int, int)
cdef float cargf_r(float complex)
cdef double cargd_r(float complex)
cdef float[::1] absmat1(float complex[::1])
cdef float[:, ::1] absmat2(float complex[:, ::1])
cdef float[::1] angmat(float complex[::1])
cdef double[::1] angmatd(float complex[::1])
cdef float[:, ::1] angmat2(float complex[:, ::1])
cdef float complex[::1] expmati(float[::1])
cdef float complex[::1] conjmat1(float complex[::1])
cdef float complex[:,::1] conjmat2(float complex[:,::1])
cdef float complex[:,::1] multiply_elementwise_dc(float[:, :], float complex[:, ::1])
cdef float complex multiplymat11(float complex[::1], float complex[::1])
cdef float complex[:,::1] multiplymat22(float complex[:, :], float complex[:, ::1])
cdef float complex[::1] multiplymat12(float complex[::1], float complex[:, ::1])
cdef float complex[::1] EVD_phase_estimation_cy(float complex[:, ::1])
cdef float complex[::1] EMI_phase_estimation_cy(float complex[:, ::1], float[:, ::1])
cdef float[::1] optimize_lbfgs(double[::1], float complex[:, ::1])
cpdef double optphase_cy(double[::1], float complex[:, ::1])
cdef float complex[::1] PTA_L_BFGS_cy(float complex[:, ::1], float[:, ::1])
cdef float[:,::1] outer_product(float[::1], float[::1])
cdef float complex[:,::1] divide_elementwise(float complex[:, ::1], float[:, ::1])
cdef float complex[:, ::1] cov2corr_cy(float complex[:,::1])
cdef float complex[:,::1] transposemat2(float complex[:, :])
cdef float complex[:,::1] est_corr_cy(float complex[:,::1])
cdef float complex[:,::1] est_cov_cy(float complex[:,::1])
cpdef float complex[:,::1] est_cov_py(float complex[:,::1])
cpdef float complex[:,::1] est_corr_py(float complex[:,::1])
cdef float sum1d(float[::1])
cdef float test_PS_cy(float complex[:, ::1], float[::1])
cdef float norm_complex(float complex[::1])
cdef float complex[::1] squeeze_images(float complex[::1], float complex[:, ::1], cnp.intp_t)
cdef tuple phase_linking_process_cy(float complex[:, ::1], int, bytes, bint, int)
cpdef tuple phase_linking_process_py(float complex[:, ::1], int, bytes, bint, int)
cpdef tuple sequential_phase_linking_py(float complex[:,::1], bytes, int, int)
cdef tuple sequential_phase_linking_cy(float complex[:,::1], bytes, int, int)
cdef float complex[::1] datum_connect_cy(float complex[:, ::1], float complex[::1], int)
cpdef float complex[::1] datum_connect_py(float complex[:, ::1], float complex[::1], int)
cdef float searchsorted_max(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1])
cdef void sorting(cnp.ndarray[float, ndim=1])
cdef float ecdf_distance(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1])
cdef float ks_lut_cy(int, int, float)
cdef cnp.ndarray[float, ndim=1] concat_cy(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1])
cdef int count(cnp.ndarray[long, ndim=2], long)
cdef int[:, ::1] get_shp_row_col_c((int, int), float complex[:, :, ::1], cnp.ndarray[int, ndim=1], cnp.ndarray[int, ndim=1],
                                   int, int, int, int, float, bytes)
cdef float[::1] mean_along_axis_x(float[:, ::1])
cdef float gam_pta_c(float[:, ::1], float complex[::1])
cdef int ks2smapletest_cy(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1], float)
cdef int ttest_indtest_cy(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1], float)
cdef int ADtest_cy(cnp.ndarray[float, ndim=1], cnp.ndarray[float, ndim=1], float)
cpdef float[:, :] inverse_float_matrix(float[:, ::1])
cdef float complex[:, ::1] normalize_samples(float complex[:, ::1])
cdef int is_semi_pos_def_chol_cy(float[:, ::1])
cdef tuple regularize_matrix_cy(float[:, ::1])
cdef float complex[:, ::1] mask_diag(float complex[:, ::1], int)
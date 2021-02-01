cimport numpy as cnp
cimport cython


cdef void write_wrapped(list, bytes, int, int, bytes, bytes)
cdef void write_hdf5_block_3D(object, float complex[:, :, ::1], bytes, list)
cdef void write_hdf5_block_2D(object, float[:, ::1], bytes, list)

ctypedef bint (*shptestptr)(float[::1], float[::1], float)

cdef class CPhaseLink:
    cdef object inps, slcStackObj
    cdef bytes work_dir, phase_linking_method, shp_test
    cdef bytes slc_stack, RSLCfile
    cdef int range_window, azimuth_window, patch_size, n_image, width, length
    cdef int shp_size, mini_stack_default_size, num_box, total_num_mini_stacks
    cdef float distance_thresh
    cdef bint mpi_flag, sequential
    cdef dict metadata
    cdef list all_date_list
    cdef int[::1] sample_rows, sample_cols
    cdef int reference_row, reference_col
    cdef float complex[:, :, ::1] patch_slc_images
    cdef shptestptr shp_function
    cdef readonly list box_list
    cdef readonly bytes out_dir



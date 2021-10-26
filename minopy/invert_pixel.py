#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee                                    #
############################################################
import os
import numpy as np
from minopy.objects.slcStack import slcStack
import minopy.lib.utils as iut
from scipy import linalg as LA
from scipy.linalg import lapack as lap
from skimage.measure import label


def ks_lut_cy(N1, N2, alpha):
    N = (N1 * N2) / (N1 + N2)
    distances = np.arange(0.01, 1, 0.001, dtype=np.float32)
    critical_distance = 0.1
    for i in range(distances.shape[0]):
        value = distances[i]*(np.sqrt(N) + 0.12 + 0.11/np.sqrt(N))
        pvalue = 0
        for t in range(1, 101):
            pvalue += ((-1)**(t-1))*np.exp(-2*(value**2)*(t**2))
        pvalue = 2 * pvalue
        if pvalue > 1:
            pvalue = 1
        if pvalue < 0:
            pvalue = 0
        if pvalue <= alpha:
            critical_distance = distances[i]
            break
    return critical_distance


def sorting(x):
    x.sort()
    return

def get_shp_row_col_c(data, input_slc, def_sample_rows, def_sample_cols,
                      reference_row, reference_col, distance_threshold):


    n_image = input_slc.shape[0]

    ref = np.zeros(n_image, dtype=np.float32)

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
        if temp >= length:
            t2 = i
            break
    s_rows = t2 - t1
    ref_row = reference_row - t1

    sample_rows = np.zeros(s_rows, dtype=np.int32)
    for i in range(s_rows):
        sample_rows[i] = row_0 + def_sample_rows[i + t1]

    t1 = 0
    t2 = def_sample_cols.shape[0]
    for i in range(def_sample_cols.shape[0]):
        temp = col_0 + def_sample_cols[i]
        if temp < 0:
            t1 += 1
        if temp >= width:
            t2 = i
            break
    s_cols = t2 - t1
    ref_col = reference_col - t1

    sample_cols = np.zeros((s_cols), dtype=np.int32)
    for i in range(s_cols):
        sample_cols[i] = col_0 + def_sample_cols[i + t1]

    for i in range(n_image):
        ref[i] = np.abs(input_slc[i, row_0, col_0])

    sorting(ref)
    distance = np.zeros((s_rows, s_cols), dtype='long')

    for t1 in range(s_rows):
        for t2 in range(s_cols):
            test = np.zeros(n_image, dtype=np.float32)
            for temp in range(n_image):
                test[temp] = np.abs(input_slc[temp, sample_rows[t1], sample_cols[t2]])
            sorting(test)
            distance[t1, t2] = iut.ks2smapletest_py(ref, test, distance_threshold)

    ks_label = label(distance, connectivity=2)
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


def count(x, value):
    n1 = x.shape[0]
    n2 = x.shape[1]
    out = 0
    for i in range(n1):
        for t in range(n2):
            if x[i, t] == value:
                out += 1
    return out

def is_semi_pos_def_chol_cy(x):
    """ Checks the positive semi definitness of a matrix. desired: res=0 """
    try:
        LA.cholesky(x)
        res = 0
    except:
        res = 1
    return res

def regularize_matrix_cy(M):
    """ Regularizes a matrix to make it positive semi definite. """
    status = 1
    t = 0
    N = np.zeros((M.shape[0], M.shape[1]), dtype=np.float32)
    en = 1e-6

    for i in range(M.shape[0]):
        for t in range(M.shape[1]):
            N[i, t] = M[i, t]

    t = 0
    while t < 100:
        status = is_semi_pos_def_chol_cy(N)
        if status == 0:
            break
        else:

            for i in range(M.shape[0]):
                N[i, i] += en
            en *= 2
            t += 1
    return status, N

def test_PS_cy(coh_mat, amplitude):
    """ checks if the pixel is PS """

    nn = coh_mat.shape[0]
    #amplitude_diff = np.zeros(nn, dtype=np.float32)

    stat, AM = regularize_matrix_cy(np.abs(coh_mat))
    coh_mat_r = AM * np.exp(1j*np.angle(coh_mat))
    Eigen_value, Eigen_vector = lap.cheevx(coh_mat_r)[0:2]

    s = 0
    for i in range(nn):
        s += abs(Eigen_value[i])**2
        #amplitude_diff[i] = amplitude[i]-amplitude[0]

    s = np.sqrt(s)
    #amp_diff_dispersion = np.std(amplitude_diff)/np.mean(amplitude)
    amp_dispersion = np.std(amplitude)/np.mean(amplitude)
    import pdb; pdb.set_trace()
    #if Eigen_value[nn-1]*(100 / s) > 25 and amp_diff_dispersion <= 0.6:
    if Eigen_value[nn-1]*(100 / s) > 25 and amp_dispersion < 0.25:
        #x0 = cexpf(1j * cargf_r(Eigen_vector[0, n-1]))
        #for i in range(n):
        #    vec[i] = Eigen_vector[i, n-1] * conjf(x0)

        temp_quality = 1 #gam_pta_c(angmat2(coh_mat), vec)
    else:
        temp_quality = 0

    return temp_quality



def process_pixel(coord, slc_stack, range_window=19, azimuth_window=9, phase_linking_method=b'sequential'):

    default_mini_stack_size = 10
    sample_rows = np.arange(-((azimuth_window - 1) // 2), ((azimuth_window - 1) // 2) + 1, dtype=np.int32)
    reference_row = np.array([(azimuth_window - 1) // 2], dtype=np.int32)
    sample_cols = np.arange(-((range_window - 1) // 2), ((range_window - 1) // 2) + 1, dtype=np.int32)
    reference_col = np.array([(range_window - 1) // 2], dtype=np.int32)

    slcStackObj = slcStack(slc_stack)
    n_image, length, width = slcStackObj.get_size()
    distance_threshold = ks_lut_cy(n_image, n_image, 0.01)
    box = [coord[1] - 100, coord[0] - 100, coord[1] + 100, coord[0] + 100]
    row1 = box[1]
    row2 = box[3]
    col1 = box[0]
    col2 = box[2]
    lin = np.arange(row1, row2, dtype=np.int32)
    overlap_length = row2 - row1
    sam = np.arange(col1, col2, dtype=np.int32)
    overlap_width = col2 - col1
    data = (coord[0] - row1, coord[1] - col1)
    patch_slc_images = slcStackObj.read(datasetName='slc', box=box, print_msg=False)
    vec_refined = np.empty(n_image, dtype=np.complex64)
    amp_refined = np.zeros(n_image, dtype=np.float32)
    total_num_mini_stacks = n_image // default_mini_stack_size
    import pdb; pdb.set_trace()
    shp = get_shp_row_col_c(data, patch_slc_images, sample_rows, sample_cols,
                            reference_row, reference_col, distance_threshold)

    num_shp = shp.shape[0]
    CCG = np.zeros((n_image, num_shp), dtype=np.complex64)
    for t in range(num_shp):
        for m in range(n_image):
            CCG[m, t] = patch_slc_images[m, shp[t, 0], shp[t, 1]]

    import pdb; pdb.set_trace()

    coh_mat = iut.est_corr_py(CCG)

    temp_quality = 0
    if num_shp < 20:
        x0 = np.conj(patch_slc_images[0, data[0], data[1]])
        for m in range(n_image):
            vec_refined[m] = patch_slc_images[m, data[0], data[1]] * x0
            amp_refined[m] = np.abs(patch_slc_images[m, data[0], data[1]])
        temp_quality = test_PS_cy(coh_mat, amp_refined)
        temp_quality_full = temp_quality

    else:

        if len(phase_linking_method) > 10 and phase_linking_method[0:10] == b'sequential':
            vec_refined, squeezed_images, temp_quality = iut.sequential_phase_linking_py(CCG, phase_linking_method,
                                                                                         default_mini_stack_size,
                                                                                         total_num_mini_stacks)

            vec_refined = iut.datum_connect_py(squeezed_images, vec_refined, default_mini_stack_size)

        else:
            vec_refined, noval, temp_quality = iut.phase_linking_process_py(CCG, 0, phase_linking_method, False, lag=10)

        amp_refined = np.mean(np.abs(CCG), axis=1)
        temp_quality_full = gam_pta_c(angmat2(coh_mat), vec_refined)

    for m in range(n_image):

        if m == 0:
            vec_refined[m] = amp_refined[m] + 0j
        else:
            vec_refined[m] = amp_refined[m] * np.exp(1j * np.angle(vec_refined[m]))


    return vec_refined, temp_quality_full, temp_quality


'''
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import numpy as np
from minopy.invert_pixel import process_pixel
coord = (659, 5998)
coord = (767, 1770)
slc_stack = './inputs/slcStack.h5'
process_pixel(coord, slc_stack)
'''
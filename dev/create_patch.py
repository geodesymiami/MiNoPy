#!/usr/bin/env python3
# Author: Sara Mirzaee


import os
import sys
import numpy as np
import time
from minopy.objects.arg_parser import MinoPyParser
from minopy_utilities import read_image, patch_slice, log_message
import minopy.submit_jobs as js
import h5py

#######################


def main(iargs=None):
    """
        Divides the whole scene into patches for parallel processing
    """
    Parser = MinoPyParser(iargs, script='create_patch')
    inps = Parser.parse()

    job_file_name = 'create_patch'
    job_name = job_file_name

    if inps.wall_time == 'None':
        inps.wall_time = '2:00'

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:
        js.submit_script(job_name, job_file_name, sys.argv[:], inps.work_dir, inps.wall_time)
        sys.exit(0)

    if not iargs is None:
        log_message(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        log_message(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    slc_file = os.path.join(inps.work_dir, 'inputs/slcStack.h5')

    inps.patch_dir = os.path.join(inps.work_dir, 'patches')

    f = h5py.File(slc_file, 'r')
    ds = f['slc']
    numDate, length, width = ds.shape

    if not os.path.exists(inps.patch_dir):
        os.mkdir(inps.patch_dir)

    patch_rows, patch_cols, patch_list = patch_slice(length, width, inps.azimuth_window,
                                                         inps.range_window, inps.patch_size)

    np.save(inps.patch_dir + '/rowpatch.npy', patch_rows)
    np.save(inps.patch_dir + '/colpatch.npy', patch_cols)

    if os.path.isfile(inps.patch_dir + '/flag.npy'):
        print('patchlist exist')
    else:

        start_time = time.time()
        for patch in patch_list:
            patch_row, patch_col = patch.split('_')
            patch_row, patch_col = (int(patch_row), int(patch_col))
            patch_name = inps.patch_dir + '/patch' + str(patch_row) + '_' + str(patch_col)

            # fcol, frow, lcol, lrow:
            box = [patch_cols[0][0][patch_col], patch_rows[0][0][patch_row],
                   patch_cols[1][0][patch_col], patch_rows[1][0][patch_row]]

            line = box[3] - box[1]
            sample = box[2] - box[0]

            if not os.path.isfile(patch_name + '/count.npy'):
                if not os.path.isdir(patch_name):
                    os.mkdir(patch_name)
                rslc = np.memmap(patch_name + '/rslc', dtype=np.complex64, mode='w+', shape=(numDate, line, sample))
                rslc[:, :, :] = ds[:, box[1]:box[3], box[0]:box[2]]

                del rslc

                np.save(patch_name + '/count.npy', [numDate, line, sample])
                print(os.path.basename(patch_name) + " is created")

    np.save(inps.patch_dir + '/flag.npy', 'patchlist_created')
    f.close()
    timep = time.time() - start_time

    print('All patches created in {} seconds'.format(timep))

    return


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing.

    '''
    main()

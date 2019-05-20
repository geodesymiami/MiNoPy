#!/usr/bin/env python3
# Author: Sara Mirzaee

import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import minopy_utilities as pysq
from rinsar.utils.process_utilities import create_or_update_template
from rinsar.objects.auto_defaults import PathFind


pathObj = PathFind()
#######################

def create_parser():
    """ Creates command line argument parser object. """
    parser = argparse.ArgumentParser(description='Divides the whole scene into patches for parallel processing')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings.\n')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def create_patch(name):
    patch_row, patch_col = name.split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))
    patch_name = pathObj.patch_dir + str(patch_row) + '_' + str(patch_col)

    line = pathObj.patch_rows[1][0][patch_row] - pathObj.patch_rows[0][0][patch_row]
    sample = pathObj.patch_cols[1][0][patch_col] - pathObj.patch_cols[0][0][patch_col]


    if not os.path.isfile(patch_name + '/count.npy'):
        if not os.path.isdir(patch_name):
            os.mkdir(patch_name)

        rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='w+', shape=(pathObj.n_image, line, sample))

        count = 0

        for dirs in pathObj.list_slv:
            data_name = pathObj.slave_dir + '/' + dirs + '/' + dirs + '.slc'
            slc = np.memmap(data_name, dtype=np.complex64, mode='r', shape=(pathObj.lin, pathObj.sam))

            rslc[count, :, :] = slc[pathObj.patch_rows[0][0][patch_row]:pathObj.patch_rows[1][0][patch_row],
                                pathObj.patch_cols[0][0][patch_col]:pathObj.patch_cols[1][0][patch_col]]
            count += 1
            del slc

        del rslc

        np.save(patch_name + '/count.npy', [pathObj.n_image,line,sample])
    else:
        print('Next patch...')
    return "PATCH" + str(patch_row) + '_' + str(patch_col) + " is created"


########################################################


def main(iargs=None):
    """
        Divides the whole scene into patches for parallel processing
    """

    inps = command_line_parse(iargs)
    inps = create_or_update_template(inps)
    inps.squeesar_dir = os.path.join(inps.work_dir, pathObj.squeesardir)
    pathObj.patch_dir = inps.squeesar_dir + '/PATCH'

    pathObj.slave_dir = os.path.join(inps.work_dir, pathObj.mergedslcdir)
    
    pathObj.list_slv = os.listdir(pathObj.slave_dir)
    pathObj.list_slv = [datetime.strptime(x, '%Y%m%d') for x in pathObj.list_slv]
    pathObj.list_slv = np.sort(pathObj.list_slv)
    pathObj.list_slv = [x.strftime('%Y%m%d') for x in pathObj.list_slv]

    inps.range_window = int(inps.range_window)
    inps.azimuth_window = int(inps.azimuth_window)

    if not os.path.isdir(inps.squeesar_dir):
        os.mkdir(inps.squeesar_dir)

    slc = pysq.read_image(pathObj.slave_dir + '/' + pathObj.list_slv[0] + '/' + pathObj.list_slv[0] + '.slc')  #
    pathObj.n_image = len(pathObj.list_slv)
    pathObj.lin = slc.shape[0]
    pathObj.sam = slc.shape[1]
    del slc

    pathObj.patch_rows, pathObj.patch_cols, inps.patch_list = \
        pysq.patch_slice(pathObj.lin, pathObj.sam, inps.azimuth_window, inps.range_window, np.int(inps.patch_size))

    np.save(inps.squeesar_dir + '/rowpatch.npy', pathObj.patch_rows)
    np.save(inps.squeesar_dir + '/colpatch.npy', pathObj.patch_cols)

    time0 = time.time()
    if os.path.isfile(inps.squeesar_dir + '/flag.npy'):
        print('patchlist exist')
    else:

        futures = []
        start_time = time.time()

        for patch in inps.patch_list:
            future = client.submit(create_patch, patch, retries=3)
            futures.append(future)

        i_future = 0
        for future, result in as_completed(futures, with_results=True):
            i_future += 1
            print("FUTURE #" + str(i_future), "complete in", time.time() - start_time, "seconds.")

    np.save(inps.squeesar_dir + '/flag.npy', 'patchlist_created')
    timep = time.time() - time0

    print("Done Creating PATCH. time:{} min".format(timep/60))

    cluster.close()
    client.close()


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing.

    '''

    try:
        from dask.distributed import Client, as_completed
        # dask_jobqueue is needed for HPC.
        # PBSCluster (similar to LSFCluster) should also work out of the box
        from dask_jobqueue import LSFCluster
        python_executable_location = sys.executable
        # Look at the ~/.config/dask/dask_mintpy.yaml file for Changing the Dask configuration defaults
        cluster = LSFCluster(config_name='create_patch',
                             python=python_executable_location)
        NUM_WORKERS = 40
        cluster.scale(NUM_WORKERS)
        print("JOB FILE:", cluster.job_script())
        # This line needs to be in a function or in a `if __name__ == "__main__":` block. If it is in no function
        # or "main" block, each worker will try to create its own client (which is bad) when loading the module
        client = Client(cluster)

    except:

        from distributed import Client, LocalCluster

        cluster = LocalCluster()
        client = Client(cluster)

    main()

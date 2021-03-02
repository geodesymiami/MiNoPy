#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

#mpl_logger = logging.getLogger('matplotlib')
#mpl_logger.setLevel(logging.WARNING)

fiona_logger = logging.getLogger('fiona')
fiona_logger.propagate = False

import time
import numpy as np

from minopy.objects.arg_parser import MinoPyParser
#from mpi4py import MPI
from minopy.lib import utils as iut
from minopy.lib import invert as iv
from math import ceil
import multiprocessing as mp
from functools import partial
#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''


    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    inversionObj = iv.CPhaseLink(inps)

    if inps.unpatch_flag:
        inversionObj.unpatch()

    else:

        box_list = []
        for box in inversionObj.box_list:
            index = box[4]
            out_dir = inversionObj.out_dir.decode('UTF-8')
            out_folder = out_dir + '/PATCHES/PATCH_{}'.format(index)
            os.makedirs(out_folder, exist_ok=True)
            if not os.path.exists(out_folder + '/quality.npy'):
                box_list.append(box)

        num_cores = np.min([mp.cpu_count(), int(inps.num_worker)])
        pool = mp.Pool(processes=num_cores)
        data_kwargs = inversionObj.get_datakwargs()
        func = partial(iut.process_patch_c, range_window=data_kwargs['range_window'],
                       azimuth_window=data_kwargs['azimuth_window'], width=data_kwargs['width'],
                       length=data_kwargs['length'], n_image=data_kwargs['n_image'],
                       slcStackObj=data_kwargs['slcStackObj'], distance_threshold=data_kwargs['distance_threshold'],
                       reference_row=data_kwargs['reference_row'],reference_col=data_kwargs['reference_col'],
                       phase_linking_method=data_kwargs['phase_linking_method'],
                       total_num_mini_stacks=data_kwargs['total_num_mini_stacks'],
                       default_mini_stack_size=data_kwargs['default_mini_stack_size'],
                       shp_test=data_kwargs['shp_test'],
                       def_sample_rows=data_kwargs['def_sample_rows'],
                       def_sample_cols=data_kwargs['def_sample_cols'])
        pool.map(func, box_list)
        pool.close()
        pool.join()

        '''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        np.random.seed(seed=rank)

        if size > len(box_list):
            num = 1
        else:
            num = ceil(len(box_list) // size) + 1

        index = np.arange(0, len(box_list), num)
        index[-1] = len(box_list)

        if rank < len(index) - 1:
            time_passed = inversionObj.loop_patches(box_list[index[rank]:index[rank+1]])
            comm.gather(time_passed, root=0)

        MPI.Finalize()
        '''

    return None


#################################################


if __name__ == '__main__':
    main()

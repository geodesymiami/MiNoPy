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
from mpi4py import MPI
from minopy.lib import utils as iut
from minopy.lib import invert as iv
from math import ceil
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
            index = [4]
            out_dir = inversionObj.out_dir.decode('UTF-8')
            out_folder = out_dir + '/PATCHES/PATCH_{}'.format(index)
            if not os.path.exists(out_folder + '/quality.npy'):
                box_list.append(box)

        if inps.mpi_flag:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            np.random.seed(seed=rank)

            if size > len(box_list):
                num = 1
            else:
                num = ceil(len(box_list) // size)

            index = np.arange(0, len(box_list), num)
            index[-1] = len(box_list)

            if rank < len(index):
                time_passed = inversionObj.loop_patches(box_list[index[rank]:index[rank+1]])
                comm.gather(time_passed, root=0)
        else:
            inversionObj.loop_patches(inversionObj.box_list)

        MPI.Finalize()

    return None


#################################################


if __name__ == '__main__':
    main()

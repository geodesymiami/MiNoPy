#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
import numpy as np
import minopy_utilities as mnp
import dask
import glob
from minsar.objects import message_rsmas
import minsar.utils.process_utilities as putils
from minsar.objects.auto_defaults import PathFind
import minsar.job_submission as js

pathObj = PathFind()


#################################


def main(iargs=None):
    '''
    Phase linking process.
    '''

    inps = mnp.cmd_line_parse(iargs)

    message_rsmas.log(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    inps.minopydir = os.path.join(inps.work_dir, pathObj.minopydir)
    patch_list = glob.glob(inps.minopydir + '/PATCH*')

    os.system('python -W ignore patch_inversion.py')

    run_minopy_inversion = os.path.join(pathObj.minopydir, 'run_minopy_inversion')

    with open(run_minopy_inversion, 'w') as f:
        for item in patch_list:
            cmd = 'patch_inversion.py {a0} -p {a1} \n'.format(a0=inps.customTemplateFile, a1=item)
            f.write(cmd)

    config = putils.get_config_defaults(config_file='job_defaults.cfg')

    step_name = 'patch_inversion'
    try:
        memorymax = config[step_name]['memory']
    except:
        memorymax = config['DEFAULT']['memory']

    try:
        if config[step_name]['adjust'] == 'True':
            walltimelimit = putils.walltime_adjust(config[step_name]['walltime'])
        else:
            walltimelimit = config[step_name]['walltime']
    except:
        walltimelimit = config['DEFAULT']['walltime']

    queuename = os.getenv('QUEUENAME')

    putils.remove_last_job_running_products(run_file=run_minopy_inversion)

    jobs = js.submit_batch_jobs(batch_file=run_minopy_inversion,
                                out_dir=inps.minopydir,
                                work_dir=inps.work_dir, memory=memorymax,
                                walltime=walltimelimit, queue=queuename)

    putils.remove_zero_size_or_length_error_files(run_file=run_minopy_inversion)
    putils.raise_exception_if_job_exited(run_file=run_minopy_inversion)
    putils.concatenate_error_files(run_file=run_minopy_inversion, work_dir=inps.work_dir)
    putils.move_out_job_files_to_stdout(run_file=run_minopy_inversion)

    return None


if __name__ == '__main__':
    main()


#################################################

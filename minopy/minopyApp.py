#!/usr/bin/env python3
########################
# Author: Sara Mirzaee
#######################
import logging
import warnings

warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import os
import sys
import time
import datetime
import shutil
import h5py
import re
import math
import subprocess

import minopy
import minopy.workflow
from mintpy.utils import writefile, readfile, utils as ut
from minsar.job_submission import JOB_SUBMIT
import mintpy
from mintpy.smallbaselineApp import TimeSeriesAnalysis
import minopy.minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects.slcStack import slcStack
from minopy.defaults.auto_path import autoPath, PathFind
from minopy.objects.utils import check_template_auto_value

pathObj = PathFind()
###########################################################################################
STEP_LIST = [
    'crop',
    'inversion',
    'ifgrams',
    'unwrap',
    'load_int',
    'reference_point',
    'correct_unwrap_error',
    'write_to_timeseries',
    'correct_SET',
    'correct_troposphere',
    'deramp',
    'correct_topography',
    'residual_RMS',
    'reference_date',
    'velocity',
    'geocode',
    'google_earth',
    'hdfeos5',
    'plot',]

#'email',]

##########################################################################


def main(iargs=None):
    start_time = time.time()

    Parser = MinoPyParser(iargs, script='minopy_app')
    inps = Parser.parse()
    
    if not iargs is None:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(iargs[:]))
    else:
        mut.log_message(inps.workDir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    os.chdir(inps.workDir)

    app = minopyTimeSeriesAnalysis(inps.customTemplateFile, inps.workDir, inps)
    app.startup
    if len(inps.runSteps) > 0:
        app.run(steps=inps.runSteps, plot=inps.plot)

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('Time used: {:02.0f} mins {:02.1f} secs\n'.format(m, s))
    return


def write_single_job_file(jobObj, job_name, job_file_name, command_line, work_dir=None, number_of_nodes=1, distribute=None):
    """
    Writes a job file for a single job.
    :param job_name: Name of job.
    :param job_file_name: Name of job file.
    :param command_line: Command line containing process to run.
    :param work_dir: working or output directory
    """
    import subprocess

    # get lines to write in job file
    job_file_lines = jobObj.get_job_file_lines(job_name, job_file_name, work_dir=work_dir,
                                               number_of_nodes=number_of_nodes)
    job_file_lines.append("\nfree")

    if jobObj.scheduler == 'SLURM':
        hostname = subprocess.Popen("hostname", shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
        job_file_lines.append("\nsmodule load python_cacher \n")
        job_file_lines.append("export PYTHON_IO_CACHE_CWD=0\n")
        job_file_lines.append("module load ooops\n")

        if not distribute is None:
            if 'stampede2' in hostname:
                job_file_lines.append('\nexport CDTOOL=/scratch/01255/siliu/collect_distribute\n')
            elif 'frontera' in hostname:
                job_file_lines.append('\nexport CDTOOL=/scratch1/01255/siliu/collect_distribute\n')

            job_file_lines.append('export PATH=${PATH}:${CDTOOL}/bin\n')
            job_file_lines.append('distribute.bash ' + distribute + '\n')

    job_file_lines.append("\n" + command_line + "\n")

    # write lines to .job file
    job_file_name = "{0}.job".format(job_file_name)
    with open(os.path.join(work_dir, job_file_name), "w+") as job_file:
        job_file.writelines(job_file_lines)

    return

class minopyTimeSeriesAnalysis(TimeSeriesAnalysis):
    """ Routine processing workflow for time series analysis of InSAR stacks with MiNoPy
        """

    def __init__(self, customTemplateFile=None, workDir=None, inps=None):
        super().__init__(customTemplateFile, workDir)
        self.inps = inps

        self.customTemplateFile = customTemplateFile
        self.cwd = os.path.abspath(os.getcwd())

        # 1. Go to the work directory
        # 1.1 Get workDir
        current_dir = os.getcwd()
        if not self.workDir:
            if 'minopy' in current_dir:
                self.workDir = current_dir.split('minopy')[0] + 'minopy'
            else:
                self.workDir = os.path.join(current_dir, 'minopy')
        self.workDir = os.path.abspath(self.workDir)

        # 2. Get project name
        self.project_name = None
        if self.customTemplateFile and not os.path.basename(self.customTemplateFile) == 'minopy_template.cfg':
            self.project_name = os.path.splitext(os.path.basename(self.customTemplateFile))[0]
            print('Project name:', self.project_name)
        else:
            self.project_name = os.path.dirname(self.workDir)

        self.run_dir = os.path.join(self.workDir, pathObj.rundir)
        # self.patch_dir = os.path.join(self.workDir, pathObj.patchdir)
        self.ifgram_dir = os.path.join(self.workDir, pathObj.intdir)
        self.templateFile = ''

        self.plot_sh_cmd = ''

        self.status = False
        self.azimuth_look = 1
        self.range_look = 1

    @property
    def startup(self):

        # 2.2 Go to workDir
        os.makedirs(self.workDir, exist_ok=True)
        os.chdir(self.workDir)
        print("Go to work directory:", self.workDir)

        # 3. Read templates
        # 3.1 Get default template file
        self.templateFile = mut.get_latest_template(self.workDir)
        # 3.2 read (custom) template files into dicts
        self._read_template()

        # 4. Copy the plot shell file
        sh_file = os.path.join(os.getenv('MINTPY_HOME'), 'mintpy/sh/plot_smallbaselineApp.sh')

        def grab_latest_update_date(fname, prefix='# Latest update:'):
            try:
                lines = open(fname, 'r').readlines()
                line = [i for i in lines if prefix in i][0]
                t = re.findall('\d{4}-\d{2}-\d{2}', line)[0]
                t = datetime.datetime.strptime(t, '%Y-%m-%d')
            except:
                t = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')  # a arbitrary old date
            return t

        # 1) copy to work directory (if not existed yet).
        if not os.path.isfile(os.path.basename(sh_file)):
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)

        # 2) copy to work directory (if obsolete file detected) and rename the existing one
        elif grab_latest_update_date(os.path.basename(sh_file)) < grab_latest_update_date(sh_file):
            os.system('mv {f} {f}_obsolete'.format(f=os.path.basename(sh_file)))
            print('obsolete shell file detected, renamed it to: {}_obsolete'.format(os.path.basename(sh_file)))
            print('copy {} to work directory: {}'.format(sh_file, self.workDir))
            shutil.copy2(sh_file, self.workDir)

        self.plot_sh_cmd = './' + os.path.basename(sh_file)

        self.range_look = int(self.template['MINOPY.interferograms.range_look'])
        self.azimuth_look = int(self.template['MINOPY.interferograms.azimuth_look'])

        return

    def _read_template(self):
        # read custom template, to:
        # 1) update default template
        # 2) add metadata to ifgramStack file and HDF-EOS5 file
        self.customTemplate = None
        if self.customTemplateFile:
            cfile = self.customTemplateFile
            # Copy custom template file to inputs directory for backup
            inputs_dir = os.path.join(self.workDir, 'inputs')
            if not os.path.isdir(inputs_dir):
                os.makedirs(inputs_dir)
                print('create directory:', inputs_dir)
            if ut.run_or_skip(out_file=os.path.join(inputs_dir, os.path.basename(cfile)),
                              in_file=cfile,
                              check_readable=False) == 'run':
                shutil.copy2(cfile, inputs_dir)
                print('copy {} to inputs directory for backup.'.format(os.path.basename(cfile)))

            # Read custom template
            print('read custom template file:', cfile)
            cdict = readfile.read_template(cfile)

            # correct some loose type errors
            standardValues = {'def':'auto', 'default':'auto',
                              'y':'yes', 'on':'yes', 'true':'yes',
                              'n':'no', 'off':'no', 'false':'no'
                             }
            for key, value in cdict.items():
                if value in standardValues.keys():
                    cdict[key] = standardValues[value]

            for key in ['mintpy.deramp', 'mintpy.troposphericDelay.method']:
                if key in cdict.keys():
                    cdict[key] = cdict[key].lower().replace('-', '_')

            if 'processor' in cdict.keys():
                cdict['MINOPY.load.processor'] = cdict['processor']

            # these metadata are used in load_data.py only, not needed afterwards
            # (in order to manually add extra offset when the lookup table is shifted)
            # (seen in ROI_PAC product sometimes)
            for key in ['SUBSET_XMIN', 'SUBSET_YMIN']:
                if key in cdict.keys():
                    cdict.pop(key)
            self.customTemplate = dict(cdict)

            # Update default template file based on custom template
            print('update default template based on input custom template')
            self.templateFile = ut.update_template_file(self.templateFile, self.customTemplate)
        print('read default template file:', self.templateFile)
        self.template = readfile.read_template(self.templateFile)
        auto_template_file = os.path.join(os.path.dirname(__file__), 'defaults/minopy_template_defaults.cfg')
        self.template = check_template_auto_value(self.template, auto_file=auto_template_file)
        # correct some loose setup conflicts
        if self.template['mintpy.geocode'] is False:
            for key in ['mintpy.save.hdfEos5', 'mintpy.save.kmz']:
                if self.template[key] is True:
                    self.template['mintpy.geocode'] = True
                    print('Turn ON mintpy.geocode in order to run {}.'.format(key))
                    break

        minopy_template = self.template.copy()
        for key, value in minopy_template.items():
            key2 = key.replace('minopy', 'mintpy')
            self.template[key2] = value
        return

    def run_crop(self, sname):
        """ Cropping images using crop_sentinel.py script.
        """

        os.chdir(self.workDir)

        if self.template['mintpy.subset.lalo'] == 'None' and self.template['mintpy.subset.yx'] == 'None':
            print('WARNING: No crop area given in mintpy.subset, the whole image is going to be used.')
            print('WARNING: May take days to process!')

        scp_args = '--template {}'.format(self.templateFile)
        if self.customTemplateFile:
            scp_args += ' {}'.format(self.customTemplateFile)
        if self.project_name:
            scp_args += ' --project {}'.format(self.project_name)

        print('crop_images.py ', scp_args)

        os.makedirs(self.run_dir, exist_ok=True)
        run_file_crop = os.path.join(self.run_dir, 'run_01_minopy_crop')
        run_commands = ['crop_images.py ' + scp_args]

        with open(run_file_crop, 'w+') as frun:
            frun.writelines(run_commands)

        inps = self.inps
        inps.work_dir = self.run_dir
        inps.out_dir = self.run_dir
        job_obj = JOB_SUBMIT(inps)
        job_obj.write_batch_jobs(batch_file=run_file_crop)

        if not inps.norun_flag:
            job_status = job_obj.submit_batch_jobs(batch_file=run_file_crop)

        return

    def run_phase_inversion(self, sname):
        """ Non-Linear phase inversion.
        """
        inps = self.inps
        inps.work_dir = self.run_dir
        inps.out_dir = self.run_dir
        inps.custom_template_file = self.customTemplateFile
        inps.memory = int(self.template['MINOPY.parallel.job_memory'])
        inps.wall_time = self.template['MINOPY.parallel.job_walltime']
        num_nodes = int(self.template['MINOPY.parallel.num_nodes'])
        num_workers = int(self.template['MINOPY.parallel.num_workers'])
        job_name = 'run_02_phase_inversion'
        job_file_name = 'run_02_phase_inversion_0'

        scp_args = '--workDir {a0} --rangeWin {a1} --azimuthWin {a2} --method {a3} --test {a4} ' \
                   '--patchSize {a5} --numWorker {a6}'.format(a0=self.workDir,
                                                              a1=self.template['MINOPY.inversion.range_window'],
                                                              a2=self.template['MINOPY.inversion.azimuth_window'],
                                                              a3=self.template['MINOPY.inversion.plmethod'],
                                                              a4=self.template['MINOPY.inversion.shp_test'],
                                                              a5=self.template['MINOPY.inversion.patch_size'],
                                                              a6=num_workers)

        scp_args2 = scp_args + ' --unpatch'

        command_line1 = '\n$MINOPY_HOME/minopy/phase_inversion.py {}'.format(scp_args)

        command_line2 = '\n$MINOPY_HOME/minopy/phase_inversion.py {} --unpatch'.format(scp_args)

        job_obj = JOB_SUBMIT(inps)


        if os.getenv('HOSTNAME') is None or job_obj.scheduler is None:

            scp_args = scp_args + ' --slcStack {a}'.format(a=os.path.join(self.workDir, 'inputs/slcStack.h5'))
            scp_args2 = scp_args2 + ' --slcStack {a}'.format(a=os.path.join(self.workDir, 'inputs/slcStack.h5'))
            print('phase_inversion.py ', scp_args)
            minopy.phase_inversion.main(scp_args.split())
            print('phase_inversion.py ', scp_args2)
            minopy.phase_inversion.main(scp_args2.split())

        else:
            os.makedirs(self.run_dir, exist_ok=True)
            scp_args = scp_args + ' --slcStack {a}'.format(a='/tmp/slcStack.h5')
            print('phase_inversion.py ', scp_args)
            command_line1 = command_line1 + ' --slcStack {a}'.format(a='/tmp/slcStack.h5')
            command_line2 = command_line2 + ' --slcStack {a}'.format(a='/tmp/slcStack.h5')
            command_line = command_line1 + command_line2
            job_obj.write_single_job_file(job_name, job_file_name, command_line,
                                  work_dir=self.run_dir, number_of_nodes=num_nodes,
                                  distribute=os.path.join(self.workDir, 'inputs/slcStack.h5'))

        return


    def run_interferogram(self, sname):
        """ Export single reference interferograms
        """

        ifgram_dir = os.path.join(self.workDir, 'inverted/interferograms')
        ifgram_dir = ifgram_dir + '_{}'.format(self.template['MINOPY.interferograms.type'])
        os.makedirs(ifgram_dir, exist_ok='True')

        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        num_pixels = int(metadata['length']) * int(metadata['width'])

        if 'sensor_type' in metadata:
            sensor_type = metadata['sensor_type']
        else:
            sensor_type = 'tops'

        if self.template['MINOPY.interferograms.referenceDate']:
            reference_date = self.template['MINOPY.interferograms.referenceDate']
        else:
            reference_date = date_list[0]

        if self.template['MINOPY.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['MINOPY.interferograms.type'] == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = date_list.index(reference_date)

        pairs = []
        if reference_ind == 'multi':
            indx = date_list.index(reference_date)
            for i in range(0, len(date_list)):
                if not indx == i:
                    pairs.append((date_list[indx], date_list[i]))
                if not i == 0:
                    pairs.append((date_list[i - 1], date_list[i]))
        else:
            for i in range(0, len(date_list)):
                if not reference_ind is None:
                    if not reference_ind == i:
                        pairs.append((date_list[reference_ind], date_list[i]))
                else:
                    if not i == 0:
                        pairs.append((date_list[i - 1], date_list[i]))
        pairs = list(set(pairs))
        # if reference_ind is False:
        #    pairs.append((date_list[0], date_list[-1]))


        inps = self.inps
        inps.run_dir = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        inps.ifgram_dir = self.ifgram_dir
        inps.template = self.template
        inps.num_bursts = num_pixels // 30000000
        run_ifgs = os.path.join(inps.run_dir, 'run_03_minopy_ifgrams')
        run_commands = []

        inps.work_dir = inps.run_dir
        inps.out_dir = inps.run_dir
        inps.custom_template_file = self.customTemplateFile
        job_obj = JOB_SUBMIT(inps)

        rslc_ref = os.path.join(self.workDir, 'inverted/rslc_ref.h5')

        if os.getenv('HOSTNAME') is None or job_obj.scheduler is None:
            write_job = False
            del job_obj
        else:
            write_job = True

        num_cpu = os.cpu_count()
        num_lin = 0
        for pair in pairs:
            out_dir = os.path.join(ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            scp_args = '--reference {a1} --secondary {a2} --outdir {a3} --alks {a4} ' \
                       '--rlks {a5} --filterStrength {a6} ' \
                       '--prefix {a7} --stack {a8}'.format(a1=pair[0],
                                                           a2=pair[1],
                                                           a3=out_dir, a4=self.azimuth_look,
                                                           a5=self.range_look,
                                                           a6=self.template['MINOPY.interferograms.filter_strength'],
                                                           a7=sensor_type,
                                                           a8=rslc_ref)

            cmd = 'generate_interferograms.py ' + scp_args
            if write_job is False:
                cmd = cmd + ' &\n'
                run_commands.append(cmd)
                num_lin += 1
                if num_lin == num_cpu:
                    run_commands.append('wait\n\n')
                    num_lin = 0
            else:
                cmd = cmd + '\n'
                run_commands.append(cmd)

        run_commands.append('wait\n\n')

        with open(run_ifgs, 'w+') as frun:
            frun.writelines(run_commands)

        if write_job:
            job_obj.write_batch_jobs(batch_file=run_ifgs)
            del job_obj
        else:
            status = subprocess.Popen(run_commands, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

        return

    def run_unwrap(self, sname):
        """ Unwrapps single reference interferograms
        """
        slc_file = os.path.join(self.workDir, 'inputs/slcStack.h5')
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        length = int(metadata['LENGTH'])
        width = int(metadata['WIDTH'])
        wavelength = metadata['WAVELENGTH']
        earth_radius = metadata['EARTH_RADIUS']
        height = metadata['HEIGHT']
        num_pixels = length * width

        if self.template['MINOPY.interferograms.referenceDate']:
            reference_date = self.template['MINOPY.interferograms.referenceDate']
        else:
            reference_date = date_list[0]

        if self.template['MINOPY.interferograms.type'] == 'sequential':
            reference_ind = None
        elif self.template['MINOPY.interferograms.type']  == 'combine':
            reference_ind = 'multi'
        else:
            reference_ind = date_list.index(reference_date)

        pairs = []
        if reference_ind == 'multi':
            indx = date_list.index(reference_date)
            for i in range(0, len(date_list)):
                if not indx == i:
                    pairs.append((date_list[indx], date_list[i]))
                if not i == 0:
                    pairs.append((date_list[i - 1], date_list[i]))
        else:
            for i in range(0, len(date_list)):
                if not reference_ind is None:
                    if not reference_ind == i:
                        pairs.append((date_list[reference_ind], date_list[i]))
                else:
                    if not i == 0:
                        pairs.append((date_list[i - 1], date_list[i]))
        pairs = list(set(pairs))

        # if reference_ind is False:
        #    pairs.append((date_list[0], date_list[-1]))

        inps = self.inps
        inps.run_dir = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        inps.ifgram_dir = self.ifgram_dir
        inps.ifgram_dir = inps.ifgram_dir + '_{}'.format(self.template['MINOPY.interferograms.type'])
        inps.template = self.template
        run_file_unwrap = os.path.join(self.run_dir, 'run_04_minopy_un-wrap')
        run_commands = []

        inps.work_dir = inps.run_dir
        inps.out_dir = inps.run_dir
        inps.custom_template_file = self.customTemplateFile
        inps.num_bursts = num_pixels // 3000000
        print('num bursts: {}'.format(inps.num_bursts))
        job_obj = JOB_SUBMIT(inps)

        if os.getenv('HOSTNAME') is None or job_obj.scheduler is None:
            write_job = False
            del job_obj
        else:
            write_job = True

        num_cpu = os.cpu_count()
        ntiles = num_pixels / 2000000
        if ntiles > 1:
            x_tile = int(math.sqrt(ntiles)) + 1
            y_tile = x_tile
        else:
            x_tile = 1
            y_tile = 1
        num_cpu = min([num_cpu, x_tile * y_tile])

        num_lin = 0

        for pair in pairs:
            out_dir = os.path.join(inps.ifgram_dir, pair[0] + '_' + pair[1])
            os.makedirs(out_dir, exist_ok='True')

            #if self.azimuth_look * self.range_look > 1:
            #    corr_file = os.path.join(self.workDir, 'inverted/quality_ml')
            #else:
            #    corr_file = os.path.join(self.workDir, 'inverted/quality')

            corr_file = os.path.join(out_dir, 'filt_fine.cor')

            scp_args = '--ifg {a1} --cor {a2} --unw {a3} --defoMax {a4} --initMethod {a5} ' \
                       '--ref_length {a6} --ref_width {a7} --height {a8} --num_tiles {a9} ' \
                       '--earth_radius {a10} --wavelength {a11}'.format(a1=os.path.join(out_dir, 'filt_fine.int'),
                                                                       a2=corr_file,
                                                                       a3=os.path.join(out_dir, 'filt_fine.unw'),
                                                                       a4=self.template['MINOPY.unwrap.defomax'],
                                                                       a5=self.template['MINOPY.unwrap.init_method'],
                                                                       a6=length, a7=width, a8=height, a9= num_cpu,
                                                                       a10=earth_radius, a11=wavelength)
            cmd = 'unwrap_minopy.py ' + scp_args

            if write_job is False:
                cmd = cmd + ' &\n'
                run_commands.append(cmd)
                num_lin += 1
                if num_lin == num_cpu:
                    run_commands.append('wait\n\n')
                    num_lin = 0
            else:
                cmd = cmd + '\n'
                run_commands.append(cmd)

            # print(cmd)
            run_commands.append(cmd)

        run_commands.append('wait\n\n')

        with open(run_file_unwrap, 'w+') as frun:
            frun.writelines(run_commands)

        if write_job:

            job_obj.write_batch_jobs(batch_file=run_file_unwrap, num_cores_per_task=num_cpu)
            del job_obj
        else:
            status = subprocess.Popen(run_commands, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

        return

    def run_load_int(self, step_name):
        """Load InSAR stacks into HDF5 files in ./inputs folder.
        It 1) copy auxiliary files into work directory (for Unvi of Miami only)
           2) load all interferograms stack files into mintpy/inputs directory.
           3) check loading result
           4) add custom metadata (optional, for HDF-EOS5 format only)
        """
        os.chdir(self.workDir)

        # 1) copy aux files (optional)
        self.projectName = self.project_name
        super()._copy_aux_file()

        # 2) loading data
        scp_args = '--template {}'.format(self.templateFile)
        if self.customTemplateFile:
            scp_args += ' {}'.format(self.customTemplateFile)
        if self.projectName:
            scp_args += ' --project {}'.format(self.projectName)
        scp_args += ' --output {}'.format(self.workDir + '/inputs/ifgramStack.h5')
        # run
        print("load_int.py", scp_args)
        minopy.load_int.main(scp_args.split())

        # 3) check loading result
        load_complete, stack_file, geom_file = ut.check_loaded_dataset(self.workDir, print_msg=True)[0:3]

        # 4) add custom metadata (optional)
        if self.customTemplateFile:
            print('updating {}, {} metadata based on custom template file: {}'.format(
                os.path.basename(stack_file),
                os.path.basename(geom_file),
                os.path.basename(self.customTemplateFile)))
            # use ut.add_attribute() instead of add_attribute.py because of
            # better control of special metadata, such as SUBSET_X/YMIN
            ut.add_attribute(stack_file, self.customTemplate)
            ut.add_attribute(geom_file, self.customTemplate)

        # 5) if not load_complete, plot and raise exception
        if not load_complete:
            # plot result if error occured
            self.plot_result(print_aux=False, plot='True')

            # go back to original directory
            print('Go back to directory:', self.cwd)
            os.chdir(self.cwd)

            # raise error
            msg = 'step {}: NOT all required dataset found, exit.'.format(step_name)
            raise RuntimeError(msg)
        return

    def run_reference_point(self, step_name):
        """Select reference point.
        It 1) generate mask file from common conn comp
           2) generate average spatial coherence and its mask
           3) add REF_X/Y and/or REF_LAT/LON attribute to stack file
        """
        self.run_network_modification(step_name)
        self.generate_ifgram_aux_file()

        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        coh_file = 'avgSpatialCoh.h5'

        scp_args = '{} -t {} -c {} --method maxCoherence'.format(stack_file, self.templateFile, coh_file)
        print('reference_point.py', scp_args)
        mintpy.reference_point.main(scp_args.split())
        self.run_quick_overview(step_name)

        return

    def write_to_timeseries(self, sname):
        if self.azimuth_look * self.range_look > 1:
            self.template['quality_file'] = os.path.join(self.workDir, 'inverted/quality_ml')
        else:
            self.template['quality_file'] = os.path.join(self.workDir, 'inverted/quality')
        mut.invert_ifgrams_to_timeseries(self.template, self.inps, self.workDir, writefile)
        functions = [mintpy.generate_mask, readfile, ut.run_or_skip, ut.add_attribute]
        mut.get_phase_linking_coherence_mask(self.template, self.workDir, functions)

        return

    def write_correction_job(self, sname):
        run_commands = ['minopyApp.py {} --start load_int'.format(self.templateFile)]
        os.makedirs(self.run_dir, exist_ok=True)
        run_file_corrections = os.path.join(self.run_dir, 'run_05_mintpy_corrections')

        with open(run_file_corrections, 'w+') as frun:
            frun.writelines(run_commands)

        inps = self.inps
        inps.custom_template_file = self.customTemplateFile
        inps.work_dir = self.run_dir
        inps.out_dir = self.run_dir
        job_obj = JOB_SUBMIT(inps)
        job_obj.write_batch_jobs(batch_file=run_file_corrections)

        return

    def run(self, steps=STEP_LIST, plot=True):
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))

            if sname == 'crop':
                self.run_crop(sname)

            elif sname == 'inversion':
                self.run_phase_inversion(sname)

            elif sname == 'ifgrams':
                self.run_interferogram(sname)

            elif sname == 'unwrap':
                self.run_unwrap(sname)

            elif sname == 'write_correction_job':
                self.write_correction_job(sname)

            elif sname == 'load_int':
                self.run_load_int(sname)

            elif sname == 'reference_point':

                ifgram_file = os.path.join(self.workDir, 'inputs/ifgramStack.h5')
                with h5py.File(ifgram_file, 'a') as f:
                    f.attrs['mintpy.reference.yx'] = self.template['mintpy.reference.yx']
                    f.attrs['mintpy.reference.lalo'] = self.template['mintpy.reference.lalo']
                f.close()

                self.run_reference_point(sname)

            elif sname == 'correct_unwrap_error':
                super().run_unwrap_error_correction(sname)

            elif sname == 'write_to_timeseries':
                self.write_to_timeseries(sname)

            elif sname == 'correct_SET':
                super().run_solid_earth_tides_correction(sname)

            elif sname == 'correct_troposphere':
                super().run_tropospheric_delay_correction(sname)

            elif sname == 'deramp':
                super().run_phase_deramping(sname)

            elif sname == 'correct_topography':
                super().run_topographic_residual_correction(sname)

            elif sname == 'residual_RMS':
                super().run_residual_phase_rms(sname)

            elif sname == 'reference_date':
                super().run_reference_date(sname)

            elif sname == 'velocity':
                super().run_timeseries2velocity(sname)

            elif sname == 'geocode':
                super().run_geocode(sname)

            elif sname == 'google_earth':
                super().run_save2google_earth(sname)

            elif sname == 'hdfeos5':
                super().run_save2hdfeos5(sname)

            elif sname == 'plot':
                # plot result (show aux visualization message more multiple steps processing)
                print_aux = len(steps) > 1
                super().plot_result(print_aux=print_aux, plot=plot)

            #elif sname == 'email':
            #    mut.email_minopy(self.workDir)

        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n###############################################################'
        msg += '\nNormal end of Non-Linear time series processing workflow!'
        msg += '\n##############################################################'
        print(msg)
        return


###########################################################################################


if __name__ == '__main__':
    main()

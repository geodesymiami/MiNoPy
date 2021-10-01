#! /usr/bin/env python3
###############################################################################
# Project: Argument parser for minopy
# Author: Sara Mirzaee
###############################################################################
import sys
import argparse
import minopy
from minopy.defaults import auto_path
import os
import datetime
pathObj = auto_path.PathFind()

CLUSTER_LIST = ['lsf', 'pbs', 'slurm', 'local']


class MinoPyParser:

    def __init__(self, iargs=None, script=None):
        self.iargs = iargs
        self.script = script
        self.parser = argparse.ArgumentParser(description='MiNoPy scripts parser')
        commonp = self.parser.add_argument_group('General options:')
        commonp.add_argument('-v', '--version', action='store_true', help='Print software version and exit')
        #commonp.add_argument('--walltime', dest='wall_time', default='None',
        #                    help='Walltime for submitting the script as a job')
        #commonp.add_argument('--queue', dest='queue', default=None, help='Queue name')
        #commonp.add_argument('--submit', dest='submit_flag', action='store_true', help='submits job')

    def parse(self):

        if self.script == 'load_slc':
            self.parser = self.load_slc_parser()
        elif self.script == 'crop_images':
            self.parser = self.crop_images_parser()
        elif self.script == 'phase_inversion':
            self.parser = self.phase_inversion_parser()
        elif self.script == 'generate_interferograms':
            self.parser = self.generate_interferograms_parser()
        elif self.script == 'generate_mask':
            self.parser = self.generate_unwrap_mask_parser()
        elif self.script == 'unwrap_minopy':
            self.parser = self.unwrap_parser()
        elif self.script == 'phase_to_range':
            self.parser = self.phase_to_range_parser()
        elif self.script == 'minopy_app':
            self.parser, self.STEP_LIST, EXAMPLE = self.minopy_app_parser()

        inps = self.parser.parse_args(args=self.iargs)

        if self.script == 'load_slc':
            inps = self.out_load_slc(inps)

        if self.script == 'minopy_app':
            inps = self.out_minopy_app(inps, EXAMPLE)

        return inps

    def out_load_slc(self, sinps):
        inps = sinps
        DEFAULT_TEMPLATE = """template:
                    ########## 1. Load Data (--load to exit after this step)
                    {}\n
                    {}\n
                    {}\n
                    {}\n
                    """.format(auto_path.isceTopsAutoPath,
                               auto_path.isceStripmapAutoPath,
                               auto_path.roipacAutoPath,
                               auto_path.gammaAutoPath)

        if inps.template_file:
            pass
        elif inps.print_example_template:
            raise SystemExit(DEFAULT_TEMPLATE)
        else:
            self.parser.print_usage()
            print(('{}: error: one of the following arguments are required:'
                   ' -t/--template, -H'.format(os.path.basename(__file__))))
            print('{} -H to show the example template file'.format(os.path.basename(__file__)))
            sys.exit(1)

        inps.project_dir = os.path.abspath(inps.project_dir)
        inps.PROJECT_NAME = os.path.basename(inps.project_dir)
        inps.work_dir = os.path.join(inps.project_dir, 'minopy')
        os.makedirs(inps.work_dir, exist_ok=True)
        inps.out_dir = os.path.join(inps.work_dir, 'inputs')
        os.makedirs(inps.out_dir, exist_ok=True)
        inps.out_file = [os.path.join(inps.out_dir, i) for i in inps.out_file]

        return inps

    def out_minopy_app(self, sinps, EXAMPLE):
        inps = sinps
        template_file = os.path.join(os.path.dirname(minopy.__file__), 'defaults/minopyApp.cfg')
        template_file_print = os.path.join(os.path.dirname(minopy.__file__), 'defaults/minopy_mintpy_print.cfg')

        # print default template
        if inps.print_template:
            raise SystemExit(open(template_file_print, 'r').read(), )

        # print software version
        if inps.version:
            raise SystemExit(minopy.version.description)

        if (not inps.customTemplateFile
                and not os.path.isfile(os.path.basename(template_file))
                and not inps.generate_template):
            self.parser.print_usage()
            print(EXAMPLE)
            msg = "ERROR: no template file found! It requires:"
            msg += "\n  1) input a custom template file, OR"
            msg += "\n  2) there is a default template 'minopyApp.cfg' in current directory."
            print(msg)
            raise SystemExit()

        # invalid input of custom template
        if inps.customTemplateFile:
            inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)
            if not os.path.isfile(inps.customTemplateFile):
                raise FileNotFoundError(inps.customTemplateFile)

            # ignore if minopy_template.cfg is input as custom template
            if os.path.basename(inps.customTemplateFile) == os.path.basename(template_file):
                inps.templateFile = inps.customTemplateFile
                inps.customTemplateFile = None

        # check input --start/end/dostep
        inps = self.read_inps2run_steps(inps)

        return inps

    def read_inps2run_steps(self, inps):
        """read/get run_steps from input arguments."""

        # check input --start/end/dostep
        for key in ['startStep', 'endStep', 'doStep']:
            value = vars(inps)[key]
            if value and value not in self.STEP_LIST:
                if not value == 'multilook':
                    msg = 'Input step not found: {}'.format(value)
                    msg += '\nAvailable steps: {}'.format(self.STEP_LIST)
                    raise ValueError(msg)

        # ignore --start/end input if --dostep is specified
        if inps.doStep:
            inps.startStep = inps.doStep
            inps.endStep = inps.doStep

        # get list of steps to run
        idx0 = self.STEP_LIST.index(inps.startStep)
        idx1 = self.STEP_LIST.index(inps.endStep)

        if idx0 > idx1:
            msg = 'input start step "{}" is AFTER input end step "{}"'.format(inps.startStep, inps.endStep)
            raise ValueError(msg)

        inps.run_steps = self.STEP_LIST[idx0:idx1 + 1]

        # empty the step list for -g option
        if inps.generate_template:
            inps.run_steps = []

        print('-' * 50)
        # message - processing steps
        if len(inps.run_steps) > 0:
            # for single step - compact version info
            if len(inps.run_steps) == 1:
                print(minopy.version.release_description)
            else:
                print(minopy.version.logo)
            print('--RUN-at-{}--'.format(datetime.datetime.now()))
            print('Current directory: {}'.format(os.getcwd()))
            print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), inps.run_steps))
            print('Remaining steps: {}'.format(self.STEP_LIST[idx0 + 1:]))

        if not inps.generate_template:
            if inps.customTemplateFile:
                path1 = os.path.dirname(inps.customTemplateFile)
                path2 = path1 + '/minopy'
            else:
                path1 = os.path.dirname(inps.templateFile)
                path2 = path1

            if not inps.workDir:
                if path1.endswith('minopy'):
                    inps.workDir = path1
                else:
                    inps.workDir = path2

        inps.workDir = os.path.abspath(inps.workDir)

        inps.projectName = None
        if inps.customTemplateFile and not os.path.basename(inps.customTemplateFile) == 'minopyApp.cfg':
            inps.projectName = os.path.splitext(os.path.basename(inps.customTemplateFile))[0]
            print('Project name:', inps.projectName)
        else:
            inps.projectName = os.path.basename(os.path.dirname(inps.workDir))

        if not os.path.exists(inps.workDir):
            os.mkdir(inps.workDir)

        print('-' * 50)
        return inps

    @staticmethod
    def load_slc_parser():

        TEMPLATE = """template:
        ########## 1. Load Data
        ## auto - automatic path pattern for Univ of Miami file structure
        ## load_slc.py -H to check more details and example inputs.
        ## compression to save disk usage for slcStack.h5 file:
        ## no   - save   0% disk usage, fast [default]
        ## lzf  - save ~57% disk usage, relative slow
        ## gzip - save ~62% disk usage, very slow [not recommend]
        
        minopy.load.processor      = auto  #[isce,snap,gamma,roipac], auto for isceTops
        minopy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
        minopy.load.compression    = auto  #[gzip / lzf / no], auto for no.
        minopy.load.autoPath       = auto    # [yes, no] auto for no
        
        minopy.load.slcFile        = auto  #[path2slc_file]
        ##---------for ISCE only:
        minopy.load.metaFile       = auto  #[path2metadata_file], i.e.: ./reference/IW1.xml, ./referenceShelve/data.dat
        minopy.load.baselineDir    = auto  #[path2baseline_dir], i.e.: ./baselines
        ##---------geometry datasets:
        minopy.load.demFile        = auto  #[path2hgt_file]
        minopy.load.lookupYFile    = auto  #[path2lat_file], not required for geocoded data
        minopy.load.lookupXFile    = auto  #[path2lon_file], not required for geocoded data
        minopy.load.incAngleFile   = auto  #[path2los_file], optional
        minopy.load.azAngleFile    = auto  #[path2los_file], optional
        minopy.load.shadowMaskFile = auto  #[path2shadow_file], optional
        minopy.load.waterMaskFile  = auto  #[path2water_mask_file], optional
        minopy.load.bperpFile      = auto  #[path2bperp_file], optional
        
        ##---------subset (optional):
        ## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
        minopy.subset.yx           = auto    #[y0:y1,x0:x1 / no], auto for no
        minopy.subset.lalo         = auto    #[S:N,W:E / no], auto for no
        """

        EXAMPLE = """example:
          load_slc.py -t PichinchaSenDT142.template
          load_slc.py -t minopyApp.cfg
          load_slc.py -t PichinchaSenDT142.template --project_dir $SCRATCH/PichinchaSenDT142
          load_slc.py -H #Show example input template for ISCE/ROI_PAC/GAMMA products
        """

        parser = argparse.ArgumentParser(description='Saving a stack of Interferograms to an HDF5 file',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=TEMPLATE + '\n' + EXAMPLE)
        parser.add_argument('-H', dest='print_example_template', action='store_true',
                            help='Print/Show the example template file for loading.')
        parser.add_argument('-t', '--template', type=str, nargs='+',
                            dest='template_file', help='Template file with path info.')

        parser.add_argument('--project_dir', type=str, dest='project_dir',
                            help='Project directory of SLC dataset to read from')
        parser.add_argument('--processor', type=str, dest='processor',
                            choices={'isce', 'gamma', 'roipac'},
                            help='InSAR processor/software of the file (This version only supports isce)',
                            default='isce')
        parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                            help='Disable the update mode, or skip checking dataset already loaded.')
        parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                            help='Compress loaded geometry while writing HDF5 file, default: None.')
        parser.add_argument('--no_metadata_check', dest='no_metadata_check', action='store_true',
                          help='Do not check for rsc files, when running via minopyApp.py')

        parser.add_argument('-o', '--output', type=str, nargs=3, dest='out_file',
                            default=['slcStack.h5',
                                     'geometryRadar.h5',
                                     'geometryGeo.h5'],
                            help='Output HDF5 file')
        return parser

    @staticmethod
    def crop_images_parser():

        TEMPLATE = """template: 
                ##---------subset (optional):
                ## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
                mintpy.subset.yx   = auto    #[1800:2000,700:800 / no], auto for no
                mintpy.subset.lalo = auto    #[31.5:32.5,130.5:131.0 / no], auto for no
                """

        EXAMPLE = """example:
              crop_images.py -t GalapagosSenDT128.tempalte --slc_dir ./merged/SLC --geometry_dir ./merged/geom_reference 
              crop_images.py -t GalapagosSenDT128.tempalte --slc_dir ./merged/SLC --geometry_dir ./merged/geom_reference --output_dir ./merged_crop 
            """

        parser = argparse.ArgumentParser(description='Crop a subset of all input files and save to output dir given the subset in template file',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=TEMPLATE + '\n' + EXAMPLE)
        parser.add_argument('-H', dest='print_example_template', action='store_true',
                            help='Print/Show the example template file for loading.')
        parser.add_argument('-t', '--template', type=str, nargs='+',
                            dest='template_file', help='Template file with path info.')
        parser.add_argument('-s', '--slc_dir', type=str, dest='slc_dir',
                            default='./merged/SLC', help='Directory of co-registered full SLCs')
        parser.add_argument('-g', '--geometry_dir', type=str, dest='geometry_dir',
                            default='./merged/geom_reference', help='Directory of full geometry files')
        parser.add_argument('--processor', type=str, dest='processor',
                            choices={'isce', 'gamma', 'roipac'},
                            help='InSAR processor/software of the file (This version only supports isce)',
                            default='isce')
        parser.add_argument('-o', '--output_dir', type=str, dest='out_dir',
                            default='./merged_crop', help='Output directory for cropped files')
        return parser

    def phase_inversion_parser(self):
        parser = self.parser
        patch = parser.add_argument_group('Phase inversion option')
        patch.add_argument('-w', '--work_dir', type=str, dest='work_dir', help='Working directory (minopy)')
        patch.add_argument('-r', '--range_window', type=str, dest='range_window', default=15,
                           help='Range window size for shp finding')
        patch.add_argument('-a', '--azimuth_window', type=str, dest='azimuth_window', default=15,
                           help='Azimuth window size for shp finding')
        patch.add_argument('-m', '--method', type=str, dest='inversion_method', default='EMI',
                           help='Inversion method (EMI, EVD, PTA, sequential_EMI, ...)')
        patch.add_argument('-l', '--time_lag', type=int, dest='time_lag', default=10,
                           help='Time lag in case StBAS is used')
        patch.add_argument('-t', '--test', type=str, dest='shp_test', default='ks',
                           help='Shp statistical test (ks, ad, ttest)')
        patch.add_argument('-p', '--patch_size', type=str, dest='patch_size', default=200,
                           help='Azimuth window size for shp finding')
        patch.add_argument('-ms', '--mini_stack_size', type=int, dest='ministack_size', default=10,
                           help='Number of images in each mini stack')
        patch.add_argument('-s', '--slc_stack', type=str, dest='slc_stack', help='SLC stack file')
        patch.add_argument('-n', '--num_worker', dest='num_worker', type=int, default=1,
                           help='Number of parallel tasks (default: 1)')
        patch.add_argument('-i', '--index', dest='sub_index', type=str, default=None,
                           help='The list containing patches of i*num_worker:(i+1)*num_worker')


        return parser

    @staticmethod
    def generate_unwrap_mask_parser():
        parser = argparse.ArgumentParser(description='Generate unwrap mask based on shadow mask and input custom mask')
        parser.add_argument('-g', '--geometry', type=str, dest='geometry_stack', required=True,
                            help='Geometry stack file with shadowMask in the datasets')
        parser.add_argument('-m', '--mask', type=str, dest='custom_mask', default=None,
                            help='Custom mask in HDF5 format')
        parser.add_argument('-o', '--output', type=str, dest='output_mask', default=None,
                            help='Output binary mask for unwrapping with snaphu')
        parser.add_argument('-q', '--quality_type', type=str, dest='quality_type', default='full',
                            help='Temporal coherence type (full or average from mini-stacks)')
        parser.add_argument('-t', '--text_cmd', type=str, dest='text_cmd', default='',
                            help='Command before calling any script. exp: singularity run dockerimage.sif')

        return parser

    @staticmethod
    def generate_interferograms_parser():

        parser = argparse.ArgumentParser(description='Generate interferogram')
        parser.add_argument('-m', '--reference', type=str, dest='reference', required=True,
                            help='Reference image')
        parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
                            help='Secondary image')
        parser.add_argument('-t', '--stack', type=str, dest='stack_file', required=True,
                            help='Phase series stack file to read from')
        parser.add_argument('-o', '--output_dir', type=str, dest='out_dir', default='interferograms',
                            help='Prefix of output int and amp files')
        parser.add_argument('-a', '--azimuth_looks', type=int, dest='azlooks', default=1,
                            help='Azimuth looks')
        parser.add_argument('-r', '--range_looks', type=int, dest='rglooks', default=1,
                            help='Range looks')
        parser.add_argument('-f', '--filter_strength', type=float, dest='filter_strength', default=0.5,
                            help='filtering strength')
        parser.add_argument('-p', '--stack_prefix', dest='prefix', type=str, default='tops'
                            , help='ISCE stack processor: options= tops, stripmap -- default = tops')

        return parser

    @staticmethod
    def unwrap_parser():
        parser = argparse.ArgumentParser(description='Unwrap using snaphu')
        parser.add_argument('-f', '--ifg', dest='input_ifg', type=str, required=True,
                            help='Input wrapped interferogram')
        parser.add_argument('-c', '--coherence', dest='input_cor', type=str, required=True,
                            help='Input coherence file')
        parser.add_argument('-u', '--unwrapped_ifg', dest='unwrapped_ifg', type=str, required=True,
                            help='Output unwrapped interferogram')
        parser.add_argument('-m', '--mask', dest='unwrap_mask', type=str, default=None,
                            help='Output unwrapped interferogram')
        parser.add_argument('-sw', '--width', dest='ref_width', type=int, default=None,
                            help='Width of Reference .h5 file')
        parser.add_argument('-sl', '--length', dest='ref_length', type=int, default=None,
                            help='Length of .h5 file')
        parser.add_argument('-w', '--wavelength', dest='wavelength', type=str, default=None,
                            help='Wavelength')
        parser.add_argument('-ht', '--height', dest='height', type=str, default=None,
                            help='Altitude of satellite')
        parser.add_argument('-er', '--earth_radius', dest='earth_radius', type=str, default=None,
                            help='Earth Radius')
        parser.add_argument('-i', '--init_method', dest='init_method', type=str, default='MST',
                            help='Unwrap initialization algorithm (MST, MCF)')
        parser.add_argument('-d', '--max_discontinuity', dest='defo_max', type=float, default=1.2,
                            help='Maximum abrupt phase discontinuity (cycles)')
        parser.add_argument('-nt', '--num_tiles', dest='num_tiles', type=int, default=1,
                            help='Number of tiles for Unwrapping in parallel')
        parser.add_argument('--rmfilter', dest='remove_filter_flag', action='store_true',
                            help='Remove filtering after unwrap')
        parser.add_argument('--tmp', dest='copy_temp', action='store_true', help='Copy and process on tmp')

        return parser

    @staticmethod
    def phase_to_range_parser():
        parser = argparse.ArgumentParser(description='Convert phase to range time series')
        parser.add_argument('-d', '--work_dir', type=str, dest='work_dir', required=True,
                            help='Working directory (minopy)')
        parser.add_argument('-n', '--num_worker', dest='num_worker', type=int, default=1,
                           help='Number of parallel tasks (default: 1)')
        return parser

    @staticmethod
    def minopy_app_parser():

        STEP_LIST = [
            'load_slc',
            'phase_inversion',
            'generate_ifgram',
            'unwrap_ifgram',
            'load_ifgram',
            'correct_unwrap_error',
            'phase_to_range',
            'mintpy_corrections']


        STEP_HELP = """Command line options for steps processing with names are chosen from the following list:
        {}
        {}

        In order to use either --start or --step, it is necessary that a
        previous run was done using one of the steps options to process at least
        through the step immediately preceding the starting step of the current run.
        """.format(STEP_LIST[0:4], STEP_LIST[4::])

        EXAMPLE = """example: 
              minopyApp.py  <custom_template_file>            # run with default and custom templates
              minopyApp.py  -h / --help                       # help 
              minopyApp.py  -H                                # print    default template options
              # Run with --start/stop/step options
              minopyApp.py PichinchaSenDT142.template --dostep  load_slc       # run the step 'download' only
              minopyApp.py PichinchaSenDT142.template --start load_slc         # start from the step 'download' 
              minopyApp.py PichinchaSenDT142.template --stop  unwrap_ifgram    # end after step 'interferogram'
              """
        parser = argparse.ArgumentParser(description='Routine Time Series Analysis for MiNoPy',
                                         formatter_class=argparse.RawTextHelpFormatter,
                                         epilog=EXAMPLE)

        parser.add_argument('customTemplateFile', nargs='?',
                            help='Custom template with option settings.\n' +
                                 "ignored if the default minopyApp.cfg is input.")
        parser.add_argument('--dir', '--work-dir', dest='workDir', default='./',
                        help='Work directory, (default: %(default)s).')

        parser.add_argument('-g', dest='generate_template', action='store_true',
                            help='Generate default template (if it does not exist) and exit.')
        parser.add_argument('-H', dest='print_template', action='store_true',
                            help='Print the default template file and exit.')
        parser.add_argument('-v', '--version', action='store_true', help='Print software version and exit')

        parser.add_argument('--walltime', dest='wall_time', default=None,
                             help='walltime for submitting the script as a job')
        parser.add_argument('--queue', dest='queue', default=None, help='Queue name')
        parser.add_argument('--jobfiles', dest='write_job', action='store_true',
                          help='Do not run the tasks, only write job files')
        parser.add_argument('--runfiles', dest='run_flag', action='store_true', help='Create run files for all steps')
        parser.add_argument('--tmp', dest='copy_temp', action='store_true', help='Copy and process on tmp')

        step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
        step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                          help='Start processing at the named step, default: {}'.format(STEP_LIST[0]))
        step.add_argument('--end', '--stop', dest='endStep', metavar='STEP', default=STEP_LIST[-1],
                          help='End processing at the named step, default: {}'.format(STEP_LIST[-1]))
        step.add_argument('--dostep', dest='doStep', metavar='STEP',
                          help='Run processing at the named step only')


        return parser, STEP_LIST, EXAMPLE








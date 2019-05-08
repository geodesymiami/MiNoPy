#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import numpy as np
import os
import sys
import gdal
import argparse
import shutil
import isce
import isceobj
import time
import glob
from isceobj.Util.ImageUtil import ImageLib as IML
from mergeBursts import multilook
from rinsar.utils.process_utilities import create_or_update_template
from pysqsar_utilities import convert_geo2image_coord, patch_slice
from rinsar.objects.auto_defaults import PathFind

pathObj = PathFind()
##############################################################################

def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given cropping box in lat/lon (from template)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?',
                        help='custom template with option settings.\n')
    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps


def cropSLC(inpstring):

    splitstr = inpstring.split()
    input_file = splitstr[splitstr.index('--input') + 1]
    output_file = splitstr[splitstr.index('--output') + 1]
    print(input_file)
    print(output_file)
    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()
    inp_file = inp_file[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]
    del ds

    out_map = np.memmap(output_file, dtype=np.complex64, mode='write', shape=(pathObj.n_lines, pathObj.width))
    out_map[:, :] = inp_file

    IML.renderISCEXML(output_file, 1, pathObj.n_lines, pathObj.width, IML.NUMPY_type('complex64'), 'BIL')

    out_img = isceobj.createSlcImage()
    out_img.load(output_file + '.xml')
    out_img.renderVRT()
    out_img.renderHdr()

    return output_file


def cropQualitymap(inpstring):

    splitstr = inpstring.split()
    input_file = splitstr[splitstr.index('--input') + 1]
    output_file = splitstr[splitstr.index('--output') + 1]

    img = isceobj.createImage()
    img.load(input_file + '.xml')
    bands = img.bands
    data_type = IML.NUMPY_type(img.dataType)
    scheme = img.scheme

    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()
    inp_file = inp_file[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]

    if bands == 2:
        inp_file2 = ds.GetRasterBand(2).ReadAsArray()
        inp_file2 = inp_file2[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]
    del ds, img

    if not (inp_file.shape[0] == pathObj.n_lines and inp_file.shape[1] == pathObj.width):

        out_map = IML.memmap(output_file, mode='write', nchannels=bands,
                             nxx=pathObj.width, nyy=pathObj.n_lines, scheme=scheme, dataType=data_type)

        if bands == 2:
            out_map.bands[0][0::, 0::] = inp_file
            out_map.bands[1][0::, 0::] = inp_file2
        else:
            out_map.bands[0][0::, 0::] = inp_file

        IML.renderISCEXML(output_file, bands,
                          pathObj.n_lines, pathObj.width,
                          data_type, scheme)

        out_img = isceobj.createImage()
        out_img.load(output_file + '.xml')
        out_img.imageType = data_type
        out_img.renderHdr()
        out_img.renderVRT()
        try:
            out_map.bands[0].base.base.flush()
        except:
            pass

        del out_map

    return output_file


def main(inps):
    """
    Crops SLC images from Isce merged/SLC directory.
    """

    slc_list = os.listdir(os.path.join(inps.work_dir, pathObj.mergedslcdir))
    slc_list = [os.path.join(inps.work_dir, pathObj.mergedslcdir, x, x + '.slc.full') for x in slc_list]

    meta_data = pathObj.get_geom_master_lists()

    cbox = [val for val in inps.cropbox.split()]
    if len(cbox) != 4:
        raise Exception('Bbox should contain 4 floating point values')

    crop_area = np.array(
        convert_geo2image_coord(inps.geo_master, np.float32(cbox[0]), np.float32(cbox[1]),
                                np.float32(cbox[2]), np.float32(cbox[3])))

    pathObj.first_row = np.int(crop_area[0])
    pathObj.last_row = np.int(crop_area[1])
    pathObj.first_col = np.int(crop_area[2])
    pathObj.last_col = np.int(crop_area[3])

    pathObj.n_lines = pathObj.last_row - pathObj.first_row
    pathObj.width = pathObj.last_col - pathObj.first_col

    run_list_slc = []
    run_list_geo = []

    for item in slc_list:
        line = '--input {i} --output {o}'.format(i=item, o=item.split('.full')[0])
        run_list_slc.append(line)


    for item in meta_data:
        line = '--input {i} --output {o}'.format(i=os.path.join(inps.geo_master, item + '.rdr.full'),
                                                 o=os.path.join(inps.geo_master, item + '.rdr'))
        run_list_geo.append(line)

    futures = []
    start_time = time.time()
    for item in run_list_slc:
        future = client.submit(cropSLC, item, retries=3)
        futures.append(future)

    for item in run_list_geo:
        future = client.submit(cropQualitymap, item, retries=3)
        futures.append(future)

    i_future = 0
    for future, result in as_completed(futures, with_results=True):
        i_future += 1
        print("FUTURE #" + str(i_future), "complete in", time.time() - start_time, "seconds.")

    cluster.close()
    client.close()

    return


if __name__ == '__main__':
    '''
    Crop SLCs.
    '''

    inps = command_line_parse()
    inps = create_or_update_template(inps)
    inps.geo_master = os.path.join(inps.work_dir, pathObj.geomasterdir)
    import pdb; pdb.set_trace()
    try:
        from dask.distributed import Client, as_completed
        # dask_jobqueue is needed for HPC.
        # PBSCluster (similar to LSFCluster) should also work out of the box
        from dask_jobqueue import LSFCluster
        python_executable_location = sys.executable
        # Look at the ~/.config/dask/dask_pysar.yaml file for Changing the Dask configuration defaults
        cluster = LSFCluster(config_name='rsmas_insar',
                             python=python_executable_location,
                             memory=inps.dask_memory,
                             walltime=inps.dask_walltime)
        NUM_WORKERS = inps.dask_num_workers
        cluster.scale(NUM_WORKERS)
        print("JOB FILE:", cluster.job_script())
        # This line needs to be in a function or in a `if __name__ == "__main__":` block. If it is in no function
        # or "main" block, each worker will try to create its own client (which is bad) when loading the module
        client = Client(cluster)

    except:

        from distributed import Client, LocalCluster

        cluster = LocalCluster(config_name='rsmas_insar',
                             python=python_executable_location,
                             memory=inps.dask_memory,
                             walltime=inps.dask_walltime)
        client = Client(cluster)

    main(inps)

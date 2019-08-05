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
from minsar.objects import message_rsmas
from isceobj.Util.ImageUtil import ImageLib as IML
from mergeBursts import multilook
import minopy_utilities as mnp
from minsar.objects.auto_defaults import PathFind
import dask

pathObj = PathFind()
##############################################################################


def main(iargs=None):
    '''
    Crop SLCs and geometry.
    '''

    inps = mnp.cmd_line_parse(iargs)
    message_rsmas.log(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    config = putils.get_config_defaults(config_file='job_defaults.cfg')

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:
        job_file_name = 'crop_sentinel'

        if inps.wall_time == 'None':
            inps.wall_time = config[job_file_name]['walltime']

        job_name = inps.customTemplateFile.split(os.sep)[-1].split('.')[0]

        js.submit_script(job_name, job_file_name, sys.argv[:], inp.work_dir, inps.wall_time)
        sys.exit(0)

    inps.geo_master = os.path.join(inps.work_dir, pathObj.geomasterdir)
    inps.master_dir = os.path.join(inps.work_dir, pathObj.masterdir)

    slc_list = os.listdir(os.path.join(inps.work_dir, pathObj.mergedslcdir))
    slc_list = [os.path.join(inps.work_dir, pathObj.mergedslcdir, x, x + '.slc.full') for x in slc_list]

    meta_data = pathObj.get_geom_master_lists()

    cbox = [val for val in inps.cropbox.split()]
    if len(cbox) != 4:
        raise Exception('Bbox should contain 4 floating point values')

    crop_area = np.array(
        mnp.convert_geo2image_coord(inps.geo_master, inps.master_dir, np.float32(cbox[0]), np.float32(cbox[1]),
                                np.float32(cbox[2]), np.float32(cbox[3])))

    pathObj.first_row = crop_area[0]
    pathObj.last_row = crop_area[1]
    pathObj.first_col = crop_area[2]
    pathObj.last_col = crop_area[3]

    pathObj.n_lines = pathObj.last_row - pathObj.first_row
    pathObj.width = pathObj.last_col - pathObj.first_col

    run_list_slc = []
    run_list_geo = []

    for item in slc_list:
        run_list_slc.append((item, item.split('.full')[0]))

    for item in meta_data:
        run_list_geo.append((os.path.join(inps.geo_master, item + '.rdr.full'),
                                  os.path.join(inps.geo_master, item + '.rdr')))


    for item in run_list_slc:
        cropSLC(item)

    for item in run_list_geo:
        cropQualitymap(item)

    print('Done cropping images.')

    return None

##############################################################################


def cropSLC(data):
    '''crop SLC images'''

    (input_file, output_file) = data

    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]
    data_type = inp_file.dtype.type
    del ds

    out_map = np.memmap(output_file, dtype=data_type, mode='write', shape=(pathObj.n_lines, pathObj.width))
    out_map[:, :] = inp_file[:, :]

    IML.renderISCEXML(output_file, 1, pathObj.n_lines, pathObj.width, IML.NUMPY_type(str(inp_file.dtype)), 'BIL')

    out_img = isceobj.createSlcImage()
    out_img.load(output_file + '.xml')
    out_img.renderVRT()
    out_img.renderHdr()

    del inp_file, out_map

    return output_file


##############################################################################


def cropQualitymap(data):
    '''crop geometry files: lat, lon, ...'''

    (input_file, output_file) = data

    img = isceobj.createImage()
    img.load(input_file + '.xml')
    bands = img.bands
    data_type = IML.NUMPY_type(img.dataType)
    scheme = img.scheme

    ds = gdal.Open(input_file + '.vrt', gdal.GA_ReadOnly)
    inp_file = ds.GetRasterBand(1).ReadAsArray()[pathObj.first_row:pathObj.last_row, pathObj.first_col:pathObj.last_col]

    if bands == 2:
        inp_file2 = ds.GetRasterBand(2).ReadAsArray()[pathObj.first_row:pathObj.last_row,
                    pathObj.first_col:pathObj.last_col]

    del ds, img

    out_map = IML.memmap(output_file, mode='write', nchannels=bands,
                         nxx=pathObj.width, nyy=pathObj.n_lines, scheme=scheme, dataType=data_type)

    if bands == 2:
        out_map.bands[0][:, :] = inp_file[:, :]
        out_map.bands[1][:, :] = inp_file2[:, :]
    else:
        out_map.bands[0][:, :] = inp_file[:, :]

    IML.renderISCEXML(output_file, bands,
                      pathObj.n_lines, pathObj.width,
                      data_type, scheme)

    out_img = isceobj.createImage()
    out_img.load(output_file + '.xml')
    out_img.renderHdr()
    out_img.renderVRT()
    try:
        out_map.bands[0].base.base.flush()
    except:
        pass

    del out_map, inp_file
    if bands == 2:
        del inp_file2

    return output_file

##############################################################################


if __name__ == '__main__':
    main()

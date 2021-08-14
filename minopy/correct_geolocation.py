#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
import h5py
from mintpy.utils import readfile

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='Correct for geolocation shift caused by DEM error')
    parser.add_argument('-g', '--geometry', dest='geometry_file', type=str,
                        help='Geometry stack File in radar coordinate, (geometryRadar.h5)')
    parser.add_argument('-d', '--demErr', dest='dem_error_file', type=str, help='DEM error file (demErr.h5)')
    parser.add_argument('--reverse', dest='reverse', action='store_true', help='Reverse geolocation Correction')

    inps = parser.parse_args(args=iargs)
    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    key  = 'geolocation_corrected'

    f = h5py.File(inps.geometry_file, 'r+')
    keys = f.attrs.keys()

    if not key in keys or f.attrs[key] == 'no':
        status = 'run'
        print('Run geolocation correction ...')
    else:
        status = 'skip'
        print('Geolocation is already done, you may reverse it using --reverse. skip ...')

    if inps.reverse:
        if key in keys and f.attrs[key] == 'yes':
            status = 'run'
            print('Run reversing geolocation correction ...')
        else:
            status = 'skip'
            print('The file is not corrected for geolocation. skip ...')

    if status == 'run':
        az_angle = np.deg2rad(readfile.read(inps.geometry_file, datasetName='azimuthAngle')[0])
        inc_angle = np.deg2rad(readfile.read(inps.geometry_file, datasetName='incidenceAngle')[0])

        dem_error = readfile.read(inps.dem_error_file, datasetName='dem')[0]

        dx = dem_error * (1/np.tan(inc_angle)) * np.cos(az_angle) / 1110000  # converted to degree
        dy = dem_error * (1/np.tan(inc_angle)) * np.sin(az_angle) / 1110000  # converted to degree

        if inps.reverse:
            f['latitude'][:, :] = f['latitude'][:, :] - dy
            f['longitude'][:, :] = f['longitude'][:, :] - dx
            f.attrs[key] = 'no'
        else:
            f['latitude'][:, :] = f['latitude'][:, :] + dy
            f['longitude'][:, :] = f['longitude'][:, :] + dx
            f.attrs[key] = 'yes'

    f.close()

    return


if __name__ == '__main__':
    main()

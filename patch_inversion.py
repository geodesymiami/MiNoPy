#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
from numpy import linalg as LA
import numpy as np
import minopy_utilities as mnp
import dask
import pandas as pd
from scipy.stats import anderson_ksamp
from skimage.measure import label
from minsar.utils.process_utilities import create_or_update_template
from minsar.objects.auto_defaults import PathFind

pathObj = PathFind()


#################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given bounding box in lat/lon')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings.\n')
    parser.add_argument('-p', '--patch', type=str, dest='patch', required=True, help='patch directory')

    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(iargs)
    return inps


class PhaseLink:
    def __init__(self, inps):
        self.minopydir = os.path.join(inps.work_dir, pathObj.minopydir)
        self.patch_rows = np.load(os.path.join(self.minopydir, 'rowpatch.npy'))
        self.patch_cols = np.load(os.path.join(self.minopydir, 'colpatch.npy'))
        self.phase_linking_method = inps.template['minopy.plmethod']
        self.range_window = int(inps.template['minopy.range_window'])
        self.azimuth_window = int(inps.template['minopy.azimuth_window'])
        self.patch_dir = inps.patch

        count_dim = np.load(inps.patch + '/count.npy')
        self.n_image = count_dim[0]
        self.length = count_dim[1]
        self.width = count_dim[2]
        
        self.rslc = np.memmap(self.patch_dir + '/RSLC', dtype=np.complex64, mode='r',
                                 shape=(self.n_image, self.length, self.width))

        self.rslc_ref = None
        self.progress = None
        self.quality = None 

    def find_shp(self):

        if not os.path.isfile(self.patch_dir + '/SHP.pkl'):

            if self.n_image < 20:
                self.num_slc = self.n_image
            else:
                self.num_slc = 20

            time0 = time.time()

            lin = np.ogrid[0:self.length]
            sam = np.ogrid[0:self.width]
            lin, sam = np.meshgrid(lin, sam)
            coords = list(map(lambda x, y: [int(x), int(y)],
                              lin.T.reshape(self.length * self.width, 1),
                              sam.T.reshape(self.length * self.width, 1)))

            shp_df = pd.DataFrame({'ref_pixel': coords})
            shp_df.insert(1, 'pixeltype', 'default')
            shp_df.insert(2, 'rows', 'default')
            shp_df.insert(3, 'cols', 'default')

            del lin, sam

            futures = []

            for item in range(len(shp_df)):
                row_0 = shp_df.at[item, 'ref_pixel'][0]
                col_0 = shp_df.at[item, 'ref_pixel'][1]

                data = (item, row_0, col_0)

                future = dask.delayed(self.get_shp_row_col)(data)
                futures.append(future)

            results = dask.compute(*futures)

            for result in results:
                item, rr, cc, pixel_type = result
                shp_df.at[item, 'pixeltype'] = pixel_type
                shp_df.at[item, 'rows'] = np.array(rr).astype(int)
                shp_df.at[item, 'cols'] = np.array(cc).astype(int)

            shp_df.to_pickle(self.patch_dir + '/SHP.pkl')

            timep = time.time() - time0

            print('time spent to find SHPs {}: min'.format(timep / 60))

        else:

            shp_df = pd.read_pickle(self.patch_dir + '/SHP.pkl')

            print('SHP Exists ...')

        return shp_df

    def get_shp_row_col(self, data):
        item, row_0, col_0 = data
        r = np.ogrid[row_0 - ((self.azimuth_window - 1) / 2):row_0 + ((self.azimuth_window - 1) / 2) + 1]
        refr = np.array([(self.azimuth_window - 1) / 2])
        r = r[r >= 0]
        r = r[r < self.length]
        refr = refr - (self.azimuth_window - len(r))
        c = np.ogrid[col_0 - ((self.range_window - 1) / 2):col_0 + ((self.range_window - 1) / 2) + 1]
        refc = np.array([(self.range_window - 1) / 2])
        c = c[c >= 0]
        c = c[c < self.width]
        refc = refc - (self.range_window - len(c))

        x, y = np.meshgrid(r.astype(int), c.astype(int), sparse=True)
        win = np.abs(self.rslc[0:self.num_slc, x, y])
        win = mnp.trwin(win)
        testvec = win.reshape(self.num_slc, len(r) * len(c))
        ksres = np.ones(len(r) * len(c))
        S1 = np.abs(self.rslc[0:self.num_slc, row_0, col_0])
        S1 = S1.flatten()
        for m in range(len(testvec[0])):
            S2 = testvec[:, m]
            S2 = S2.flatten()

            try:
                test = anderson_ksamp([S1, S2])
                if test.significance_level > 0.05:
                    ksres[m] = 1
                else:
                    ksres[m] = 0
            except:
                ksres[m] = 0

        ks_res = ksres.reshape(len(r), len(c))
        ks_label = label(ks_res, background=False, connectivity=2)
        reflabel = ks_label[refr.astype(int), refc.astype(int)]
        rr, cc = np.where(ks_label == reflabel)
        rr = rr + r[0]
        cc = cc + c[0]

        if len(rr) > 20:
            pixel_type = 'DS'
        else:
            pixel_type = 'Unknown'

        return item, rr, cc, pixel_type

    def inversion_sequential(self, data):
        ref_row, ref_col, rr, cc = data

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(rr)).reshape(self.n_image, len(rr)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, rr, cc])

        phase_refined = mnp.sequential_phase_linking(CCG, self.phase_linking_method, num_stack=1)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        self.rslc_ref[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = \
            np.complex64(np.multiply(amp_refined, np.exp(1j * phase_refined))).reshape(self.n_image, 1, 1)

        self.progress[ref_row:ref_row + 1, ref_col:ref_col + 1] = 1

        return None

    def inversion_all(self, data):
        ref_row, ref_col, rr, cc = data

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(rr)).reshape(self.n_image, len(rr)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, rr, cc])

        phase_refined = mnp.phase_linking_process(CCG, 1, self.phase_linking_method, squeez=False)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        self.rslc_ref[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = \
            np.complex64(np.multiply(amp_refined, np.exp(1j * phase_refined))).reshape(self.n_image, 1, 1)

        self.progress[ref_row:ref_row + 1, ref_col:ref_col + 1] = 1

        return None

    def patch_phase_linking(self, shp_df):

        if not os.path.exists(self.patch_dir + '/RSLC_ref'):

            self.rslc_ref = np.memmap(inps.patch + '/RSLC_ref', dtype='complex64', mode='w+',
                                         shape=(self.n_image, self.length, self.width))

            self.progress = np.memmap(inps.patch + '/progress', dtype='int8', mode='w+',
                                         shape=(self.length, self.width))

            self.rslc_ref[:, :, :] = self.rslc[:, :, :]

        else:
            self.rslc_ref = np.memmap(inps.patch + '/RSLC_ref', dtype='complex64', mode='r+',
                                         shape=(self.n_image, self.length, self.width))

            self.progress = np.memmap(inps.patch + '/progress', dtype='int8', mode='r+',
                                         shape=(self.length, self.width))

        futures = []

        time0 = time.time()

        for item in range(len(shp_df)):

            ref_row, ref_col = (shp_df.at[item, 'ref_pixel'][0], shp_df.at[item, 'ref_pixel'][1])
            rr = shp_df.at[item, 'rows'].astype(int)
            cc = shp_df.at[item, 'cols'].astype(int)

            if self.progress[ref_row, ref_col] == 0:

                data = (ref_row, ref_col, rr, cc)

                if 'sequential' in self.phase_linking_method:
                    future = dask.delayed(inversion_sequential)(data)
                    futures.append(future)
                else:
                    future = dask.delayed(inversion_all)(data)
                    futures.append(future)

                del ref_col, ref_row

        results = dask.compute(*futures)

        timep = time.time() - time0
        print('time spent to do phase linking {}: min'.format(timep / 60))

        del self.progress

        return

    def get_pta_coherence(self, shp_df):

        if not os.path.exists(self.patch_dir + '/quality'):

            self.quality = np.memmap(self.patch_dir + '/quality', dtype='float32', mode='w+',
                                        shape=(self.length, self.width))
            self.quality[:, :] = -1

            if self.rslc_ref is None:
                self.rslc_ref = np.memmap(inps.patch + '/RSLC_ref', dtype='complex64', mode='r',
                                          shape=(self.n_image, self.length, self.width))

            futures = []

            for item in range(len(shp_df)):

                coord = (shp_df.at[item, 'ref_pixel'][0], shp_df.at[item, 'ref_pixel'][1])

                if shp_df.at[item, 'pixeltype'] == 'Unknown':
                    amp_ps = np.abs(self.rslc_ref[:, coord[0], coord[1]]).reshape(self.n_image, 1)
                    DA = np.std(amp_ps) / np.mean(amp_ps)
                    if DA < 0.25:
                        self.quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = 1
                else:

                    phase_initial = np.angle(self.rslc[:, coord[0], coord[1]]).reshape(self.n_image, 1)
                    phase_refined = np.angle(self.rslc_ref[:, coord[0], coord[1]]).reshape(self.n_image, 1)
                    data = (phase_initial, phase_refined, coord[0], coord[1])
                    
                    future = dask.delayed(mnp.gam_pta)(data)
                    futures.append(future)

            results = dask.compute(*futures)

            for result in results:
                gam, row, col = result
                self.quality[row:row + 1, col:col + 1] = gam

        else:
            print('Done with the inversion of {}'.format(self.patch_dir))

        del self.rslc_ref, self.rslc, self.quality

        return


if __name__ == '__main__':
    '''
    Phase linking process.
    '''

    inps = command_line_parse()
    inps = create_or_update_template(inps)

    inversionObj = PhaseLink(inps)

    # Find SHPs:

    shp = inversionObj.find_shp()

    # Phase linking inversion:

    inversionObj.patch_phase_linking(shp_df=shp)

    # Quality (temporal coherence) calculation based on PTA

    inversionObj.get_pta_coherence(shp_df=shp)








#################################################

#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import datetime
import isceobj
import numpy as np
from minopy.objects.arg_parser import MinoPyParser
import h5py
from math import sqrt, exp
from scipy.interpolate import griddata

enablePrint()


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MinoPyParser(iargs, script='generate_interferograms')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    os.makedirs(inps.out_dir, exist_ok=True)

    resampName = inps.out_dir + '/fine'
    resampInt = resampName + '.int'
    filtInt = os.path.dirname(resampInt) + '/filt_fine.int'
    cor_file = os.path.dirname(resampInt) + '/filt_fine.cor'

    #if os.path.exists(cor_file + '.xml'):
    #    return

    run_interferogram(inps, resampName)

    filter_strength = inps.filter_strength
    runFilter(resampInt, filtInt, filter_strength)

    estCoherence(filtInt, cor_file)

    return


def run_interferogram(inps, resampName):
    if inps.azlooks * inps.rglooks > 1:
        extention = '.ml.slc'
    else:
        extention = '.slc'

    with h5py.File(inps.stack_file, 'r') as ds:
        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list==inps.reference)[0]
        sec_ind = np.where(date_list==inps.secondary)[0]

        phase_series = ds['phase']
        amplitudes = ds['amplitude']
        tempCoh = ds['temporalCoherence'][0, :, :]   # average
        mask_r, mask_c = np.where(tempCoh >= 0.4)
        length = phase_series.shape[1]
        width = phase_series.shape[2]

        resampInt = resampName + '.int'

        ref_phase = phase_series[ref_ind, :, :].reshape(length, width)
        sec_phase = phase_series[sec_ind, :, :].reshape(length, width)

        ref_amplitude = amplitudes[ref_ind, :, :].reshape(length, width)
        sec_amplitude = amplitudes[sec_ind, :, :].reshape(length, width)

        ifg = (ref_amplitude * sec_amplitude) * np.exp(1j * (ref_phase - sec_phase))


        z1 = np.real(ifg)[mask_r, mask_c]
        z2 = np.imag(ifg)[mask_r, mask_c]

        # target grid to interpolate to
        yi = np.arange(0, length)
        xi = np.arange(0, width)
        xi, yi = np.meshgrid(xi, yi)

        zf1 = griddata((mask_r, mask_c), z1, (yi, xi), method='linear')
        zf2 = griddata((mask_r, mask_c), z2, (yi, xi), method='linear')

        ifg_intp = zf1 + 1j * zf2

        ifg_intp[np.isnan(ifg_intp)] = ifg[np.isnan(ifg_intp)]

        intImage = isceobj.createIntImage()
        intImage.setFilename(resampInt)
        intImage.setAccessMode('write')
        intImage.setWidth(width)
        intImage.setLength(length)
        intImage.createImage()

        out_ifg = intImage.asMemMap(resampInt)
        out_ifg[:, :, 0] = ifg_intp[:, :]

        intImage.renderHdr()
        intImage.finalizeImage()

    return

def runFilter(infile, outfile, filterStrength):
    from mroipac.filter.Filter import Filter

    # Initialize the flattened interferogram
    intImage = isceobj.createIntImage()
    intImage.load( infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram', object=filtImage)
    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()

def runFilterG(infile, outfile, filterStrength):

    # Initialize the flattened interferogram
    intImage = isceobj.createIntImage()
    intImage.load(infile + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outfile)
    filtImage.setWidth(intImage.getWidth())
    filtImage.setLength(intImage.getLength())
    filtImage.setAccessMode('write')
    filtImage.createImage()

    img = intImage.memMap(mode='r', band=0)
    original = np.fft.fft2(img[:, :])
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLP(100, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)

    out_filtered = filtImage.asMemMap(outfile)
    out_filtered[:, :, 0] = inverse_LowPass[:, :]

    filtImage.renderHdr()

    intImage.finalizeImage()
    filtImage.finalizeImage()


def estCoherence(outfile, corfile):
    from mroipac.icu.Icu import Icu

    # Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.load(outfile + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType = 'FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(filtImage.getWidth())
    phsigImage.setFilename(corfile)
    phsigImage.setAccessMode('write')
    phsigImage.createImage()

    icuObj = Icu(name='sentinel_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False
    # icuObj.correlationType = 'NOSLOPE'

    icuObj.icu(intImage=filtImage, phsigImage=phsigImage)
    phsigImage.renderHdr()

    filtImage.finalizeImage()
    phsigImage.finalizeImage()


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

if __name__ == '__main__':
    main()



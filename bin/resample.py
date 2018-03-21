#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import h5py
import pandas as pd
import SimpleITK as sitk
from skimage.segmentation import clear_border
from skimage import morphology
from skimage import measure
from skimage import filters
import numpy as np
import scipy
from scipy.ndimage.morphology import binary_fill_holes


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file')
parser.add_argument('-t', dest='loop_number')
parser.add_argument('-m', dest='patient_id')
parser.add_argument('-s', dest='spacing')
parser.add_argument('-o', dest='output_file')
args = parser.parse_args()

def resample(image, old_spacing, new_spacing=[1,1,1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing



old_spacing = np.loadtxt(args.spacing)

id = np.loadtxt(args.patient_id,dtype = 'S')


f= h5py.File(args.input_file,'r')
with h5py.File(args.output_file,'w') as t:
    for j in range(20):
        sample = f[id[j]][:,:,:]
        img, spa = resample(sample, old_spacing[0], new_spacing=[1,1,1])
        t.create_dataset(id[j],data = img)
        t.create_dataset(id[j]+'spacing',data = spa)



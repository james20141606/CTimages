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
from scipy.ndimage.morphology import binary_fill_holes


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='loop_number')
parser.add_argument('-o', dest='output_file')
args = parser.parse_args()

patient_info = pd.read_csv('data/annotations.csv')
#patient path
patient_path = {}
a = int(args.loop_number)
for i in range((a-1)*20,a*20):
    patient_path[i] = 'data/train_set/'+ np.unique(np.array(patient_info["seriesuid"]))[i] + '.mhd'

spacing = {}
img = {}
for i in range((a-1)*20,a*20):
    img[i] = sitk.ReadImage(patient_path[i])
    spacing[i] = np.array(img[i].GetSpacing())

spacing = np.array([val for (key,val) in spacing.iteritems()])

np.savetxt(args.output_file,spacing,fmt='%f')


"""
{
counts=$(seq 1 30)
for count in $counts;do
echo bin/get_spacing.py \
-i ${count} \
-o preprocess/spacing_info_${count}.txt
done
} > Jobs/get_spacing.txt
qsubgen -n get_spacing -q Z-LU -a 1-6 -j 5 --bsub --task-file Jobs/get_spacing.txt
bsub < Jobs/get_spacing.sh
"""

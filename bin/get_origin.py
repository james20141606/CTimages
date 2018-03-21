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
parser.add_argument('-t', dest='output_file1')
parser.add_argument('-p', dest='output_file2')
parser.add_argument('-q', dest='output_file3')
args = parser.parse_args()

patient_info = pd.read_csv('data/annotations.csv')
#patient path
patient_path = {}
a = int(args.loop_number)
for i in range((a-1)*20,a*20):
    patient_path[i] = 'data/train_set/'+ np.unique(np.array(patient_info["seriesuid"]))[i] + '.mhd'

origin = {}
img = {}
dimension_x ={}
dimension_y ={}
dimension_z ={}
img_array = {}
for i in range((a-1)*20,a*20):
    img[i] = sitk.ReadImage(patient_path[i])
    origin[i] = np.array(img[i].GetOrigin())
    img_array[i] = sitk.GetArrayFromImage(sitk.ReadImage(patient_path[i]))
    dimension_x[i] = img_array[i].shape[0]
    dimension_y[i] = img_array[i].shape[1]
    dimension_z[i] = img_array[i].shape[2]

origin = np.array([val for (key,val) in origin.iteritems()])
dimension_x = np.array([val for (key,val) in dimension_x.iteritems()])
dimension_y = np.array([val for (key,val) in dimension_y.iteritems()])
dimension_z = np.array([val for (key,val) in dimension_z.iteritems()])

np.savetxt(args.output_file,origin,fmt='%f')
np.savetxt(args.output_file1,dimension_x,fmt='%f')
np.savetxt(args.output_file2,dimension_y,fmt='%f')
np.savetxt(args.output_file3,dimension_z,fmt='%f')

"""
{
counts=$(seq 1 30)
for count in $counts;do
echo bin/get_origin.py \
-i ${count} \
-o preprocess/origin_info_${count}.txt \
-t preprocess/dimension_x_info_${count}.txt \
-p preprocess/dimension_y_info_${count}.txt \
-q preprocess/dimension_z_info_${count}.txt
done
} > Jobs/get_origin.txt
qsubgen -n get_origin -q Z-LU -a 1-6 -j 5 --bsub --task-file Jobs/get_origin.txt
bsub < Jobs/get_origin.sh
"""

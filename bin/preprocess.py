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
parser.add_argument('-i', dest='img_array')
parser.add_argument('-t', dest='patient_id')
args = parser.parse_args()

def get_segmented_lungs(im, plot=False, THRESHOLD = -320):

    binary = im < THRESHOLD

    cleared = clear_border(binary)

    label_image = measure.label(cleared)

    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    #print areas
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    selem = morphology.disk(2)
    binary = morphology.binary_erosion(binary, selem)

    selem = morphology.disk(15)
    binary = morphology.binary_closing(binary, selem)

    edges = filters.roberts(binary)
    binary = binary_fill_holes(edges)

    get_high_vals = binary == 0
    im[get_high_vals] = 0
    
    return im

id = np.loadtxt(args.patient_id,dtype = 'S')

print 'start'

processed_array = {}
with h5py.File(args.img_array,'r') as f:
    for j in range(20):
        test = f[id[j]][:,:,:]
        tensor_dim1 = test.shape[0]
        processed_array[j] = {}
        for i in range(tensor_dim1):
            processed_array[j][i] = get_segmented_lungs(test[i], THRESHOLD = -320)

print 'array processed'

processed_tensor = {}
for i in range(20):
    processed_tensor[i] = np.array([val for (key,val) in processed_array[i].iteritems()])
    print str(i)+'th tensor processed'

with h5py.File('preprocess/processed_tensor','w') as f:
    for i in range(20):
        f.create_dataset('preprocess'+id[i],data = processed_tensor[i])
        print str(i)+'th tensor saved'




#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse, sys, os, errno
import numpy as np
import os
import csv
from glob import glob
import pandas as pd
import SimpleITK as sitk
from skimage import measure,morphology
from sklearn.cluster import KMeans
from skimage.transform import resize
import seaborn as sns
import h5py
from skimage.segmentation import clear_border
from skimage import morphology
from skimage import measure
from skimage import filters
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-t', dest='number')
args = parser.parse_args()

######################################################################################
########a series of process work,deleter the constant threshold#######################
######################################################################################
def get_segmented_lungs(im, plot=False):
    cleared = clear_border(im)

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

#两种需要处理的images preprocess/forunet/images/3/ &  preprocess/forunet/images/diameter/

filename_ = np.loadtxt('preprocess/forunet/images/diameter/a.txt',dtype='S')
#1-39
count = int(args.number)
filename = filename_[(count-1)*25:count*25]
for i in tqdm(range(filename.shape[0])):
    img_file =filename[i]
    imgs_to_process = np.load('preprocess/forunet/images/diameter/'+img_file).astype(np.float64)
    imgs_shape = imgs_to_process.shape[0]

    imgss = np.ndarray([imgs_shape,imgs_to_process.shape[1],imgs_to_process.shape[2]],dtype=np.float32)
    for j in range(imgs_shape):
######################################################################################
#######################cluster to adjust threshold####################################
######################################################################################
        img = imgs_to_process[j]
        #Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        if std <=0.01:
            img = img/(std+10**(-2))
        else:
            img = img/std
        # Find the average pixel value near the lungs
        #         to renormalize washed out images
        middle = img[100:400,100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the
        #         underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        # Using Kmeans to separate foreground (radio-opaque tissue)
        #     and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        #     the non-tissue parts of the image as much as possible
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        im =get_segmented_lungs(thresh_img, plot=False)
        imgss[j] = im
    np.save(os.path.join("preprocess/forunet/images/processed_diameter/%s" % (filename[i])),imgss)







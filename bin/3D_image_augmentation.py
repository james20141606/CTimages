#! /usr/bin/env python
import argparse, sys, os, errno
import numpy as np
import os
import csv
from glob import glob
import pandas as pd
from skimage.transform import resize
import seaborn as sns
import h5py
from tqdm import tqdm
from scipy.misc import imsave
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='img_array')
parser.add_argument('-t', dest='patient_id')
parser.add_argument('-o', dest='output_file')
args = parser.parse_args()

#3D images!  注意channel last
def augment_images(images,batch_size):
    seed =123
    labels = np.arange(images.shape[0])
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=3,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         channel_shift_range=0.2,
                         zoom_range=0.05,
                         horizontal_flip=False,
                         data_format='channels_first')
     image_datagen = ImageDataGenerator(**data_gen_args)



     image_datagen.fit(images,augment=True, seed=seed)
     X_aug = []
     Z_aug = []

     i_batch = 0
     for X, y in image_datagen.flow(images, labels, batch_size=batch_size,seed=seed):
         X_aug.append(X)
         i_batch += 1
             if i_batch >= batch_size:
                 break
    X_aug = np.vstack(X_aug)
    return X_aug

file = 'preprocess/forunet/segment/3_merge'
with h5py.File(file) as f:
    images = f['images'][:,:,:,:]
    masks = f['masks'][:,:,:,:]

aug_images = np.ndarray([images.shape[0]*10,1,512,512],dtype=np.float32)
aug_masks = np.ndarray([masks.shape[0]*10,1,512,512],dtype=np.float32)
for i in tqdm(range(images.shape[0])):
    a,b = augment_images(images[i:i+1,:,:,:],masks[i:i+1,:,:,:],10)
    for j in range(10):
        aug_images[10*i+j] = a[j]
        aug_masks[10*i+j] = b[j]

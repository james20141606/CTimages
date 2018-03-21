#! /usr/bin/env python

import argparse, sys, os, errno
import logging
import numpy as np # linear algebra
import h5py
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='img_array')
parser.add_argument('-t', dest='patient_id')
args = parser.parse_args()

with h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/augment_merge') as f:
    images = f['images'][:,:,:,:]
    masks = f['masks'][:,:,:,:]
index = np.arange(images.shape[0])
np.random.shuffle(index)
#check for nan and delete
a = []
for i in range(29130):
    if np.isnan(np.sum(images[i])):
        a.append(i)
nan = np.array(a)
index = np.setdiff1d(index,a)

train_index = {}
test_index = {}
num = int(np.floor(index.shape[0]/5.0))
for i in range(5):
    test_index[i] = index[num*i:num*(i+1)]
    train_index[i] = np.setdiff1d(index,test_index[i])
print (train_index[0].shape)
print (test_index[0].shape)

train_set_img = {}
test_set_img = {}
train_set_msk = {}
test_set_msk = {}
for i in tqdm(range(5)):
    train_set_img[i] =  images[train_index[i]]
    test_set_img[i] = images[test_index[i]]
    train_set_msk[i] =  masks[train_index[i]]
    test_set_msk[i] = masks[test_index[i]]

with h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_augment') as f:
    for i in tqdm(range(5)):
        f.create_dataset('images_train_'+str(i),data = train_set_img[i])
        f.create_dataset('images_test_'+str(i),data = test_set_img[i])
        f.create_dataset('masks_train_'+str(i),data = train_set_msk[i])
        f.create_dataset('masks_test_'+str(i),data = test_set_msk[i])




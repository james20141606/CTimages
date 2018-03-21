#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy.misc import imresize
from tqdm import tqdm
import argparse, sys, os, errno
import h5py
#未归一化，可能因此训练不好。现在逐层归一化
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='count')
args = parser.parse_args()

images_train = {}
images_test={}
masks_train={}
masks_test_true ={}
i = int(args.count)
with h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_224*224') as f:
    images_train[i] = f['images_train_'+str(i)][:,:,:,:]
    images_test[i] = f['images_test_'+str(i)][:,:,:,:]
    masks_train[i] = f['masks_train_'+str(i)][:,:,:,:]
    masks_test_true[i] =f['masks_test_'+str(i)][:,:,:,:]

min_max_scaler = preprocessing.MinMaxScaler()
def image_resize(image):
    image_ = X_train_minmax = min_max_scaler.fit_transform(image)
    return image_

images_train_ = np.ndarray([images_train[i].shape[0],1,224,224],dtype=np.float32)
images_test_= np.ndarray([images_test[i].shape[0],1,224,224],dtype=np.float32)
masks_train_ = np.ndarray([images_train[i].shape[0],1,224,224],dtype=np.float32)
masks_test_ = np.ndarray([images_test[i].shape[0],1,224,224],dtype=np.float32)
for j in tqdm(range(images_train[i].shape[0])):
    images_train_[j,0,:,:] =image_resize(images_train[i][j,0,:,:])
    masks_train_[j,0,:,:] =image_resize(masks_train[i][j,0,:,:])

for t in tqdm(range(images_test[i].shape[0])):
    images_test_[t,0,:,:] =image_resize(images_train[i][t,0,:,:])
    masks_test_[t,0,:,:] =image_resize(masks_train[i][t,0,:,:])

f =  h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_224*224_minmax')
f.create_dataset('images_train_'+str(i),data = images_train_)
f.create_dataset('images_test_'+str(i),data = images_test_)
f.create_dataset('masks_train_'+str(i),data = masks_train_)
f.create_dataset('masks_test_'+str(i),data = masks_test_)


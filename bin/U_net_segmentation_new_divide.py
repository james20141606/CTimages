#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import argparse, sys, os, errno
import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import keras as K
from keras.callbacks import TensorBoard
from multi_gpu import make_parallel
from keras.callbacks import EarlyStopping


np.random.seed(1234)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


from keras.callbacks import EarlyStopping
from train_unet_newloss_single_gpu_divide import *
model = get_unet()
#model_checkpoint = ModelCheckpoint('output/unet2.hdf5', monitor='loss', save_best_only=True)
def Model(images_train,images_test,masks_train,masks_test_true):
    model.fit(images_train, masks_train, batch_size=3, nb_epoch=500,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=50, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/pr/output/tensorboard/unet/12.7_divide/log_dir')])
    num_test = images_test.shape[0]
    masks_test = np.ndarray([num_test,1,224,224],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
        masks_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/pr/output/masksTestPredicted_12.7_divide.npy', masks_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(masks_test_true[i,0], masks_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    return mean


#准备数据
images_train = {}
images_test={}
masks_train={}
masks_test_true ={}
'''
for i in range(1):
    f =  h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_augment_minmax')
    images_train[i] = f['images_train'][:,:,:,:]
    images_test[i] = f['images_test'][:,:,:,:]
    masks_train[i] = f['masks_train'][:,:,:,:]
    masks_test_true[i] =f['masks_test'][:,:,:,:]
'''
#准备数据
images_train = {}
images_test={}
masks_train={}
masks_test_true ={}
'''
for i in range(1):
    f =  h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_224*224_minmax')
    images_train[i] = f['images_train_'+str(i)][:,:,:,:]
    images_test[i] = f['images_test_'+str(i)][:,:,:,:]
    masks_train[i] = f['masks_train_'+str(i)][:,:,:,:]
    masks_test_true[i] =f['masks_test_'+str(i)][:,:,:,:]
'''
i = 0
f =  h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_not_shuffle_minmax')
images_train[i] = f['images_train'][:,:,:,:]
images_test[i] = f['images_test'][:,:,:,:]
masks_train[i] = f['masks_train'][:,:,:,:]
masks_test_true[i] =f['masks_test'][:,:,:,:]


sample_num_train = 3000
sample_num_test = 300
mean = {}
for i in tqdm(range(1)):
    mean[i] = Model(images_train[i][:sample_num_train,:,:,:],images_test[i][:sample_num_test,:,:,:],masks_train[i][:sample_num_train,:,:,:],masks_test_true[i][:sample_num_test,:,:,:])
    with open('/home/chenxupeng/projects/pr/output/'+str(i)+'_dice.txt', 'w') as f:
        f.write('%f' % mean[i])
model.save('/home/chenxupeng/projects/pr/output/12.7unet.hdf5')


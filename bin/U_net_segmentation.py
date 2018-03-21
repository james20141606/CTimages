#! /usr/bin/env python
# -*- coding: UTF-8 -*-
from train_unet import *
import numpy as np
import argparse, sys, os, errno
import os
from train_unet import *
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
from train_unet_newloss import *

model = get_unet(2)
model_checkpoint = ModelCheckpoint('/home/chenxupeng/projects/pr/output/unet2.hdf5', monitor='loss', save_best_only=True)
model.load_weights('/home/chenxupeng/projects/pr/output/unet1.hdf5')
def Model(images_train,images_test,masks_train,masks_test_true,count):
    model.fit(images_train, masks_train, batch_size=8, nb_epoch=50,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=0),model_checkpoint,TensorBoard(log_dir='/home/chenxupeng/projects/pr/output/tensorboard/unet/'+str(count)+'_1/log_dir')])
    num_test = images_test.shape[0]
    masks_test = np.ndarray([num_test,1,224,224],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
      masks_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/pr/output/masksTestPredicted_'+str(count)+'_1.npy', masks_test)
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

for i in range(5):
    f =  h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_224*224_minmax')
    images_train[i] = f['images_train_'+str(i)][:,:,:,:]
    images_test[i] = f['images_test_'+str(i)][:,:,:,:]
    masks_train[i] = f['masks_train_'+str(i)][:,:,:,:]
    masks_test_true[i] =f['masks_test_'+str(i)][:,:,:,:]

sample_num_train = 1000
sample_num_test = 200
mean = {}
for i in tqdm(range(5)):
    mean[i] = Model(images_train[i][:sample_num_train,:,:,:],images_test[i][:sample_num_test,:,:,:],masks_train[i][:sample_num_train,:,:,:],masks_test_true[i][:sample_num_test,:,:,:],i)
    np.savetxt('home/chenxupeng/projects/pr/output/'+str(i)+'_dice.txt',mean[i],fmt='%f')


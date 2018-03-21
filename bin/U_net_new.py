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
config.gpu_options.per_process_gpu_memory_fraction = 0.99
set_session(tf.Session(config=config))


from keras.callbacks import EarlyStopping
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

def convert_to_tf(images):
    imgs = np.einsum('ijkl->iklj',images)
    return imgs

for i in range(1):
    images_train[i] = convert_to_tf(images_train[i])
    images_test[i] = convert_to_tf(images_test[i])
    masks_train[i] = convert_to_tf(masks_train[i])
    masks_test_true[i] =convert_to_tf(masks_test_true[i])

def check_bad(images):
    if np.sum(images[200:,:]) >300:
        return True
    else:
        return False

bad = []
for i in range(images_train[0].shape[0]):
    if check_bad(images_train[0][i,:,:,0]):
        bad.append(i)
number = []
for i in bad:
    number.append(np.sum(images_train[0][i,:,:,0]))
number = np.array(number)
bad_ = []
for i in range(images_test[0].shape[0]):
    if check_bad(images_test[0][i,:,:,0]):
        bad_.append(i)
number_ = []
for i in bad_:
    number_.append(np.sum(images_train[0][i,:,:,0]))
number_ = np.array(number_)

images_train[0] = np.delete(images_train[0],bad,axis = 0)
images_test[0] = np.delete(images_test[0],bad_,axis = 0)
masks_train[0] = np.delete(masks_train[0],bad,axis = 0)
masks_test_true[0] = np.delete(masks_test_true[0],bad_,axis = 0)

m = 130
n = 140
k =  135
l = 145
for i in range(1000):
    a = np.random.randint(m,n)
    b = np.random.randint(k,l)
    masks_train[0][i,a:b,a:b,0] += np.random.randint(5)
    images_train[0][i,a:b,a:b,0] += np.random.randint(5)
for i in range(500):
    a = np.random.randint(m,n)
    b = np.random.randint(k,l)
    images_test[0][i,a:b,a:b,0]+= np.random.randint(5)
    masks_test_true[0][i,a:b,a:b,0] += np.random.randint(5)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from zf_unet_224_model import *
from keras.optimizers import Adam

model = ZF_UNET_224()
#model.load_weights("zf_unet_224.h5") # optional
optim = Adam()
model.compile(optimizer=optim, loss=loss, metrics=[dice_coef,jacard_coef,dice])
#model.compile(optimizer=optim, loss=jacard_coef_loss, metrics=[dice_coef,jacard_coef])

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('output/newunet.hdf5', monitor='val_dice_coef', save_best_only=True)
def Model(images_train,images_test,masks_train,masks_test_true,count):
    model.fit(images_train, masks_train, batch_size=4, nb_epoch=1,
              verbose=1, shuffle=False,validation_split=0.1,
              callbacks=[model_checkpoint,EarlyStopping(monitor='val_dice_coef', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/pr/output/tensorboard/unet/newunet'+str(count)+'_1/log_dir')])
    num_test = images_test.shape[0]
    masks_test = np.ndarray([num_test,224,224,1],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
        masks_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/pr/output/newunet_masksTestPredicted_'+str(count)+'.npy', masks_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(masks_test_true[i,0], masks_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    model.save('output/newunet.hdf5')
    return mean

#2170 . 542  26217   2913
sample_num_train = 1000
sample_num_test = 100
mean = {}
sig={}
for i in tqdm(range(1)):
    mean[i] = Model(images_train[i][:sample_num_train,:,:,:],images_test[i][:sample_num_test,:,:,:],masks_train[i][:sample_num_train,:,:,:],masks_test_true[i][:sample_num_test,:,:,:],i)


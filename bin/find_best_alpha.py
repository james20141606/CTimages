#! /usr/bin/env python

from __future__ import print_function, division
import numpy as np
import argparse, sys, os, errno
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import keras as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

list = [0.1,0.01,0.001,0.0001,0.002,0.003,0.004,0.005,0.006,0.006,0.008,0.009,0.02,0.03,0.05,0.08,0.0003,0.0006,0.0008]



K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='count')
args = parser.parse_args()

alpha = list[int(args.count)]
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) )

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f))

#0.0001
def dice_coef_loss(y_true, y_pred):
    y, x = np.mgrid[:224, :224]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    m = y_pred /K.sum(y_pred)
    sigma_x = K.sum(K.abs(x - K.sum(x*m))*m)
    sigma_y = K.sum(K.abs(y - K.sum(y*m))*m)
    loss = - (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) ) +alpha*(K.sqrt(sigma_x)*K.sqrt(sigma_y))
    return loss

def sigma(y_true,y_pred):
    y, x = np.mgrid[:224, :224]
    y_pred_f = K.flatten(y_pred)
    m = y_pred /K.sum(y_pred)
    sigma_x = K.sum(K.abs(x - K.sum(x*m))*m)
    sigma_y = K.sum(K.abs(y - K.sum(y*m))*m)
    sigma = K.sqrt(sigma_x)*K.sqrt(sigma_y)
    return sigma

def sigma_np(y_true,y_pred):
    y, x = np.mgrid[:224, :224]
    y_pred_f = y_pred.flatten()
    m = y_pred /np.sum(y_pred)
    sigma_x = np.sum(np.abs(x - np.sum(x*m))*m)
    sigma_y = np.sum(np.abs(y - np.sum(y*m))*m)
    sigma = np.sqrt(sigma_x)*np.sqrt(sigma_y)
    return sigma

img_rows = 224
img_cols = 224
def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)

    #Adam(lr=1.0e-5)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,clipvalue=0.5)
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef,sigma])

    return model


images_train = {}
images_test={}
masks_train={}
masks_test_true ={}

i = 0
with h5py.File('/home/chenxupeng/projects/pr/preprocess/forunet/segment/cv_'+str(i)+'_augment_minmax') as f:
    images_train[i] = f['images_train'][:,:,:,:]
    images_test[i] = f['images_test'][:,:,:,:]
    masks_train[i] = f['masks_train'][:,:,:,:]
    masks_test_true[i] =f['masks_test'][:,:,:,:]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.callbacks import EarlyStopping
model = get_unet()
#model_checkpoint = ModelCheckpoint('output/unet_newloss.hdf5', monitor='dice_coef', save_best_only=True)
def Model(images_train,images_test,masks_train,masks_test_true,count):
    model.fit(images_train, masks_train, batch_size=16, nb_epoch=50,
              verbose=1, shuffle=False,validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_dice_coef', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/pr/output/tensorboard/unet/'+str(count)+'_1/log_dir')])
    num_test = images_test.shape[0]
    masks_test = np.ndarray([num_test,1,224,224],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
        masks_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/pr/output/masksTestPredicted_'+str(count)+'_1.npy', masks_test)
    mean = 0.0
    sig = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(masks_test_true[i,0], masks_test[i,0])
        sig +=sigma_np(masks_test_true[i,0],masks_test[i,0])
    mean/=num_test
    sig /=num_test
    print("Mean Dice Coeff : ",mean)
    return mean,sig

#2170 . 542  26217   2913
sample_num_train = 1000
sample_num_test = 100
i = 0
mean_,sig_ = Model(images_train[i][:sample_num_train,:,:,:],images_test[i][:sample_num_test,:,:,:],masks_train[i][:sample_num_train,:,:,:],masks_test_true[i][:sample_num_test,:,:,:],i)

array = np.concatenate(([mean_],[sig_]))
np.savetxt('output/find_alpha_'+str(alpha),array,fmt ='%f')





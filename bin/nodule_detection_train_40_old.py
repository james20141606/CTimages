#! /usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import time
import numpy as np
import h5py
import random



def conv_layers(net_in):
    """ conv1 """
    network = Conv3dLayer(net_in, act = tf.nn.relu, shape = [3, 3, 3, 1, 32],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv1_1')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 32, 32],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool3d, name ='pool1')
    """ conv2 """
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 32, 64],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv2_1')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 64, 64],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                        padding='SAME', pool=tf.nn.max_pool3d, name='pool2')
    """ conv3 """
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 64, 128],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv3_1')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 128, 128],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv3_2')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 128, 128],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv3_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                        padding='SAME', pool=tf.nn.max_pool3d, name='pool3')
    """ conv4 """
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 128, 256],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv4_1')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 256, 256],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv4_2')
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 256, 256],
                strides = [1, 1, 1, 1, 1], padding='SAME', name ='conv4_3')
    """ conv5 """
    network = Conv3dLayer(network, act = tf.nn.relu, shape = [3, 3, 3, 256, 256],
                          strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5_1')
    network = Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 256, 256],
                          strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5_2')
    network = Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 256, 256],
                          strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5_3')
    return network



def fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')
    network = DenseLayer(network, n_units=1024, act=tf.identity, name='fc2_relu')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop3')
    network = DenseLayer(network, n_units=2, act=tf.identity, name='fc3_relu')
    return network


def main():

    f = h5py.File('preprocess/3D/all_40/all_40_augment')   #75
    names = f.keys()
    posiname = []
    neganame = []
    for name in names:
        if name[:4] == 'posi':
            posiname.append(name)
        elif name[:4] == 'nega':
            neganame.append(name)

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    posi_train = posiname[:2000]
    nega_train = neganame[:2000]
    names_train = posi_train + nega_train

    for train_name in names_train:
        X_train.append(f[train_name])
        if train_name[:4] == 'posi':
            y_train.append(1)
        elif train_name[:4] == 'nega':
            y_train.append(0)

    posi_val = posiname[2000:]
    nega_val = neganame[2000:]
    names_val = posi_val + nega_val

    for val_name in names_val:
        X_val.append(f[val_name])
        if val_name[:4] == 'posi':
            y_val.append(1)
        elif val_name[:4] == 'nega':
            y_val.append(0)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)


    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_train.shape', X_val.shape)
    print('y_train.shape', y_val.shape)

    sess = tf.InteractiveSession()

    batch_size = 128      #set to a proper number according to the memory

    x = tf.placeholder(tf.float32, [batch_size, 40, 40, 40, 1])
    y_ = tf.placeholder(tf.int64, shape=[batch_size, ], name='y_')


    net_in = InputLayer(x, name='input')
    net_cnn = conv_layers(net_in)
    network = fc_layers(net_cnn)
    y = network.outputs

    l2 = 0
    for w in get_variables_with_name('W_conv3d', train_only=True, printable=False):
        l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)

    cost = tl.cost.cross_entropy(y, y_, name='cost')+l2
    yBinary = tf.argmax(y, 1)

    total = tf.constant(batch_size, dtype=tf.int64)
    one = tf.constant(1, dtype=tf.int64)

    TP = tf.reduce_sum(tf.multiply(yBinary,y_)) #True positive
    FN = tf.reduce_sum(y_) - TP                    #false negative
    TN = tf.reduce_sum(tf.multiply((one-yBinary),(one-y_))) #True negative
    FP = total - tf.reduce_sum(y_) - TN
    TP = tf.cast(TP, tf.float32)
    FN = tf.cast(FN, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)

    true_positive_rate = TP/tf.maximum(TP+FN, 1.0)
    true_negative_rate = TN/tf.maximum(TN+FP, 1.0)
    false_positive_rate = FP/tf.maximum(FP+TN, 1.0)
    false_negative_rate = FN/tf.maximum(FN+TP, 1.0)
    precision = TP/tf.maximum(TP+FP, 1.0)
    accuracy = (TP+TN)/(TP+FN+TN+FP)

    tf.summary.scalar("true_positive_rate", true_positive_rate)
    tf.summary.scalar("true_negative_rate", true_negative_rate)
    tf.summary.scalar("false_positive_rate", false_positive_rate)
    tf.summary.scalar("false_negative_rate", false_negative_rate)
    tf.summary.scalar("precision", precision)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("cost", cost)


    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("train", graph=sess.graph)
    test_writer = tf.summary.FileWriter("test", graph=sess.graph)

    n_epoch = 50
    learning_rate = 0.0001
    print_freq = 1

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                # err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                summary = sess.run(merged, feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch)


            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                # err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                summary = sess.run(merged, feed_dict=feed_dict)
                test_writer.add_summary(summary, epoch)


    tl.files.save_npz(network.all_params , name='train/model_40.npz')

main()

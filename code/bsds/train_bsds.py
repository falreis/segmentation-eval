from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import datetime

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from itertools import product
from functools import partial
#K.set_image_data_format('channels_first')

import numpy as np
import json
import argparse
import h5py
np.random.seed(7) # for reproducibility

import constants as const

sys.path.append("..")
import load_weights as lw

#parameters
batch_size = 4

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def Vote(y_true, y_pred):
    vote_value = const.vote_value - 0.5
    sig_pred = K.hard_sigmoid((y_pred-vote_value)*5)
    return categorical_crossentropy(y_true, sig_pred)

def w_categorical_crossentropy(y_true, y_pred, weights):
    return K.categorical_crossentropy(y_pred, y_true) * K.cast(weights[:,:],K.floatx())

def train(model, net, merge='max', check=True, load=True, nb_epoch=100, learn_rate=0.001, folder=None):
    if(net != None):
        print(datetime.datetime.now())


        train_data = np.load('../data/HED-BSDS/train_data.npy')
        train_label = np.load('../data/HED-BSDS/train_label.npy')

        # define files
        checkpoint_file = ''
        if(folder != None and folder != ''):
            checkpoint_file = '../weights/{}/{}_bsds_weight_{}.best.hdf5'.format(folder, net, merge)
            checkpoint_file_ope = '../weights/{}/{}_bsds_weight_{}_ope.best.hdf5'.format(folder, net, merge)
        else:
            checkpoint_file = '../weights/{}_bsds_weight_{}.best.hdf5'.format(net, merge)
            checkpoint_file_ope = '../weights/{}_bsds_weight_{}_ope.best.hdf5'.format(net, merge)
            
        #load weights
        if(load):
            lw.load_weights_from_hdf5_group_by_name(model, '../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        sgd = SGD(lr=learn_rate, decay=5e-6, momentum=0.95, nesterov=False)

        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy", ofuse_pixel_error])  #metrics={'ofuse': ofuse_pixel_error})

        # Fit the model
        if(check):
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            checkpoint_ope = ModelCheckpoint(checkpoint_file_ope, monitor='val_ofuse_pixel_error', save_best_only=True, mode='min')

            callbacks_list = [checkpoint, checkpoint_ope]

            model.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                            verbose=2, shuffle=True, validation_split=0.20)
        else:
            model.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
                            verbose=2, shuffle=True, validation_split=0.20)

        print(datetime.datetime.now())

    else:
        print('Incorrect usage. Please read README.')

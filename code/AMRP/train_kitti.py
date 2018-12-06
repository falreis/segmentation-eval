from __future__ import absolute_import
from __future__ import print_function
import os
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
from keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
#K.set_image_data_format('channels_first')

import numpy as np
import json
import argparse
import h5py
import load_weights as lw
np.random.seed(7) 

import hed_constants as hc

#parameters
nb_epoch = 400
batch_size = 10

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def Vote(y_true, y_pred):
    vote_value = hc.vote_value - 0.5
    sig_pred = K.hard_sigmoid((y_pred-vote_value)*5)
    return categorical_crossentropy(y_true, sig_pred)

def train(net, merge='max', vote=0, check=True, augm=True, load=True):
    if(net != None):
        print(datetime.datetime.now())

        # load the data
        augm_str = ''
        if(augm):
            augm_str = '_augm'

        train_data = np.load('../data/Kitti/train_data{}.npy'.format(augm_str))
        train_label = np.load('../data/Kitti/train_label{}.npy'.format(augm_str))

        # define files
        if(net == 'full'):
            json_model = '../model-json/full_kitti_model.json'
            checkpoint_file = '../weights/full_kitti_weight.best.hdf5'

        else: #hed
            if(vote != None and vote > 0):
                json_model = '../model-json/hed_kitti_model_maj_{}.json'.format(vote)
                checkpoint_file = '../weights/hed_kitti_weight_maj_{}.best.hdf5'.format(net, vote)
            else:
                json_model = '../model-json/hed_kitti_model_{}.json'.format( merge)
                checkpoint_file = '../weights/hed_kitti_weight_{}.best.hdf5'.format(merge)
        #endif

        #load model
        #print(json_model)
        with open(json_model) as model_file:
            net_basic = models.model_from_json(model_file.read())
            
        #load weights
        if(load):
            lw.load_weights_from_hdf5_group_by_name(net_basic, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=False)

        if(merge == 'maj'):
            hc.vote_value = vote
            net_basic.compile(loss=Vote, optimizer=sgd, metrics=["accuracy"]) #metrics={'ofuse': ofuse_pixel_error})
        else:
            net_basic.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]) #metrics={'ofuse': ofuse_pixel_error})

        # Fit the model
        if(check):
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            history = net_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                            verbose=2, shuffle=True, validation_split=0.15)
        else:
            history = net_basic.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
                            verbose=2, shuffle=True, validation_split=0.15)

        print(datetime.datetime.now())

    else:
        print('Incorrect usage. Please read README.')

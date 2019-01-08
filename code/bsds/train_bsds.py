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
from numpy.random import shuffle
import json
import argparse
import h5py

import random
import cv2
import imutils
np.random.seed(7) # for reproducibility

#import constants
import constants as const  

#impor helper
sys.path.append("..")
import load_weights as lw
from helper import *

#parameters

batch_size = 4
steps_epochs = 100
validation_steps = steps_epochs / 10
val_split = 0.8

def loadListFile():
    data_path = '../datasets/HED-BSDS/'
    filepath =  data_path + 'train_pair.lst'
    datafile = open(filepath, 'r')
    lines = datafile.readlines() 
    datafile.close()

    shuffle(lines)
    images, grounds = [], []

    for line in lines:
        image_file, ground_file = line.replace('\n','').split(' ')
        images.append(data_path + image_file)
        grounds.append(data_path + ground_file)

    return images, grounds

def gen(images, grounds, batch_size, is_train, split):
    len_data = len(images)
    split_len = int(len_data * split)

    reduced_image_size = (const.width, const.height, 3)
    features_size = (1, 3, const.height, const.width)
    labels_size = (1, const.data_shape, const.n_classes)

    while True:
        batch_features = np.zeros(features_size)
        batch_labels = np.zeros(labels_size)

        for i in range(1): #batch_size):
            if(is_train):
                index= random.randint(0, split_len)
            else:
                index= random.randint(split_len, len_data)

            image = cv2.imread(images[index])
            ground = cv2.imread(grounds[index])

            if(image.shape[0] > image.shape[1]):
                image = imutils.rotate_bound(image, 90)
                ground = imutils.rotate_bound(ground, 90)

            reduced_image = cv2.resize(image, dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
            reduced_ground = cv2.resize(ground, dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)

            batch_features[i] = np.rollaxis(normalized(reduced_image), 2)
            batch_labels[i] = gen_hot_bsds(reduced_ground, height = const.height, width = const.width, classes = const.n_classes)

        yield (batch_features, batch_labels)

def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def train(model, net, merge='max', check=True, load=True, nb_epoch=100, learn_rate=0.001, folder=None):
    if(net != None):
        print(datetime.datetime.now())

        # define folder and files
        checkpoint_file = ''
        if(folder != None and folder != ''):
            checkpoint_file = '../weights/bsds/{}/{}_bsds_weight_{}.best.hdf5'.format(folder, net, merge)
            checkpoint_file_ope = '../weights/bsds/{}/{}_bsds_weight_{}_ope.best.hdf5'.format(folder, net, merge)
        else:
            checkpoint_file = '../weights/bsds/{}_bsds_weight_{}.best.hdf5'.format(net, merge)
            checkpoint_file_ope = '../weights/bsds/{}_bsds_weight_{}_ope.best.hdf5'.format(net, merge)
        
        checkpoint_folder = '../weights/bsds/{}/'.format(folder)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        #load weights
        if(load):
            lw.load_weights_from_hdf5_group_by_name(model, '../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

        sgd = SGD(lr=learn_rate, decay=5e-6, momentum=0.95, nesterov=False)

        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy", ofuse_pixel_error])

        # Fit the model
        images, grounds = loadListFile()

        if(check):
            checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
            checkpoint_ope = ModelCheckpoint(checkpoint_file_ope, monitor='val_ofuse_pixel_error', verbose=0, save_best_only=True, mode='min')

            callbacks_list = [checkpoint, checkpoint_ope]

            model.fit_generator(generator=gen(images, grounds, batch_size, True, val_split), steps_per_epoch=steps_epochs,
                                validation_data=gen(images, grounds, batch_size, False, val_split), validation_steps=validation_steps,
                                callbacks=callbacks_list, nb_epoch=nb_epoch, verbose=2)
        else:
            model.fit_generator(generator=gen(images, grounds, batch_size, True, val_split), steps_per_epoch=steps_epochs,
                                validation_data=gen(images, grounds, batch_size, False, val_split), validation_steps=validation_steps,
                                batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

        print(datetime.datetime.now())

    else:
        print('Incorrect usage. Please read README.')

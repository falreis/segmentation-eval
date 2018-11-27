from __future__ import absolute_import
from __future__ import print_function
import os
import datetime

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
import argparse

print(datetime.datetime.now())

#import contants
import hed_constants as hc
height = hc.height
width = hc.width
data_shape =  hc.data_shape
n_classes = hc.n_classes

#parser
import hed_args_run as ha

#parameters
nb_epoch = 100
batch_size = 5

if(ha.net_parse != None):
    # load the data
    train_data = np.load('../data/Kitti/train_data.npy')
    train_label = np.load('../data/Kitti/train_label.npy')

    # define files
    print('Vote value: ', ha.vote_value)
    if(ha.vote_value > 0):
        json_model = '../model-json/hed_kitti_model_{}_{}.json'.format(ha.merge_name, ha.vote_value)
        #hdf5_file = '../weights/{}_kitti_weight_{}_{}.hdf5'.format(ha.net_parse, ha.merge_name, ha.vote_value)
        checkpoint_file = '../weights/{}_kitti_weight_{}_{}.best.hdf5'.format(ha.net_parse, ha.merge_name, ha.vote_value)

    elif(ha.out_value > 0):
        json_model = '../model-json/hed_kitti_model_{}_{}.json'.format(ha.merge_name, ha.out_value)
        #hdf5_file = '../weights/{}_kitti_weight_{}_{}.hdf5'.format(ha.net_parse, ha.merge_name, ha.out_value)
        checkpoint_file = '../weights/{}_kitti_weight_{}_{}.best.hdf5'.format(ha.net_parse, ha.merge_name, ha.out_value)

    else:
        json_model = '../model-json/{}_kitti_model_{}.json'.format(ha.net_parse, ha.merge_name)
        #hdf5_file = '../weights/{}_kitti_weight_{}.hdf5'.format(ha.net_parse, ha.merge_name)
        checkpoint_file = '../weights/{}_kitti_weight_{}.best.hdf5'.format(ha.net_parse, ha.merge_name)
    #endif

    #load model
    with open(json_model) as model_file:
        net_basic = models.model_from_json(model_file.read())

    net_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])       


    # Fit the model
    if(ha.check_value):
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = net_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, shuffle=True, validation_split=0.20)
    else:
        history = net_basic.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, shuffle=True, validation_split=0.20)


    #save model
    #net_basic.save_weights(hdf5_file)
    print(datetime.datetime.now())

else:
    print('Incorrect usage. Please read README.')
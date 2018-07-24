from __future__ import absolute_import
from __future__ import print_function
import os
import datetime

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

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
np.random.seed(7) # 0bserver07 for reproducibility

print(datetime.datetime.now())

#parser
parser = argparse.ArgumentParser()
parser.add_argument("--net", type = str)
args = parser.parse_args()
net_parse = args.net

#shape
height = 184 #~375/2
width = 616 #~1242/2
data_shape = height*width
n_classes = 2

#parameters
nb_epoch = 20
batch_size = 3

# load the data
train_data = np.load('./data/Kitti/train_data.npy')
train_label = np.load('./data/Kitti/train_label.npy')

# define parameters
json_model, weights_file = "", ""
if net_parse == "unet":
    json_model = 'uNet_kitti_model.json'
    weights_file = "unet_kitti_weights.best.hdf5"
else:
    json_model = 'segNet_kitti_model.json'
    weights_file = "segnet_kitti_weights.best.hdf5"

#load model
with open(json_model) as model_file:
    net_basic = models.model_from_json(model_file.read())

net_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
history = net_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, shuffle=True, validation_split=0.20)

# This save the trained model weights to this file with number of epochs
if net_parse == "unet":
    net_basic.save_weights('weights/segnet_model_kitti_{}.hdf5'.format(nb_epoch))
else:
    net_basic.save_weights('weights/unet_model_kitti_{}.hdf5'.format(nb_epoch))

print(datetime.datetime.now())


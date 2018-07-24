from __future__ import absolute_import
from __future__ import print_function
import os
import datetime

os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

print(datetime.datetime.now())

data_shape = 184*616

# load the data
train_data = np.load('./data/Kitti/train_data.npy')
train_label = np.load('./data/Kitti/train_label.npy')

# load the model:
with open('uNet_kitti_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
filepath="unet_kitti_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

nb_epoch = 50 #100
batch_size = 2 #6

# Fit the model
history = segnet_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, shuffle=True, validation_split=0.33)

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('weights/unet_model_kitti_{}.hdf5'.format(nb_epoch))

print(datetime.datetime.now())


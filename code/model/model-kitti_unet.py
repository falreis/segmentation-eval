from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import keras.models as models
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate, Input

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

height = 184
width = 616
data_shape = height*width
classes = 2

inputs = Input((3, height, width))

conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Dropout(0.2)(conv1)
conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv1)

conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Dropout(0.2)(conv2)
conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv2)

conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Dropout(0.2)(conv3)
conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)

up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1)
conv4 = Dropout(0.2)(conv4)
conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv4)

up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up2)
conv5 = Dropout(0.2)(conv5)
conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv5)

conv6 = Convolution2D(classes, (1, 1), activation='relu',padding='same')(conv5)
conv6 = Reshape((classes, data_shape), input_shape=(classes, height, width))(conv6)
conv6 = Permute((2, 1))(conv6)

conv7 = Activation('softmax')(conv6)

#model = Model(input=inputs, output=conv7)
model = Model(inputs=inputs, outputs=conv7)

# Save model to JSON
with open('uNet_kitti_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
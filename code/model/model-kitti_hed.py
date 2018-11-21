from __future__ import absolute_import
from __future__ import print_function
import os

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import keras.models as models
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Average, Concatenate, Input, Conv2DTranspose

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

height = 192
width = 624
data_shape =  height*width
classes = 2

def side_branch(classes, x, factor):
    x = Convolution2D(classes, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(classes, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    #x = Conv2DTranspose(classes, kernel_size, strides=factor, use_bias=False, activation=None)(x)

    return x


###########
# NETWORK #
inputs = Input((3, height, width))

# Block 1
x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
b1= side_branch(classes, x, 1) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 240 240 64

# Block 2
x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
b2= side_branch(classes, x, 2) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128

# Block 3
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
b3= side_branch(classes, x, 4) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256

# Block 4
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
b4= side_branch(classes, x, 8) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512

# Block 5
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
b5= side_branch(classes, x, 16) # 480 480 1

# fuse
#fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
fuse = Average()([b1, b2, b3, b4, b5])
fuse = Convolution2D(classes, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

# outputs
'''
o1    = Activation('sigmoid', name='o1')(b1)
o2    = Activation('sigmoid', name='o2')(b2)
o3    = Activation('sigmoid', name='o3')(b3)
o4    = Activation('sigmoid', name='o4')(b4)
o5    = Activation('sigmoid', name='o5')(b5)
ofuse = Activation('sigmoid', name='ofuse')(fuse)
'''

#reshape
ofuse = Reshape((classes, data_shape), input_shape=(classes, height, width))(fuse)
ofuse = Permute((2, 1))(ofuse)
out = Activation('softmax')(ofuse)

model = Model(inputs=inputs, outputs=ofuse)
#print(model.summary())

# Save model to JSON
with open('model-json/hed_kitti_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

height = 184
width = 616
data_shape = height*width
classes = 2

def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, (kernel, kernel), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_first"),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, (kernel, kernel), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size),  data_format="channels_first"),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, (kernel, kernel), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, (kernel, kernel), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, (kernel, kernel), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, (kernel, kernel), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, (kernel, kernel), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, (kernel, kernel), padding='valid'),
        BatchNormalization(),
    ]

segnet_basic = models.Sequential()

segnet_basic.add(Layer(input_shape=(3, height, width)))

segnet_basic.encoding_layers = create_encoding_layers()
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)

segnet_basic.decoding_layers = create_decoding_layers()
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)

segnet_basic.add(Convolution2D(classes , (1, 1), padding='valid'))

#print(segnet_basic.output_shape)

segnet_basic.add(Reshape((classes, data_shape), input_shape=(classes, height, width)))
segnet_basic.add(Permute((2, 1)))
segnet_basic.add(Activation('softmax'))

# Save model to JSON

with open('segNet_kitti_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))
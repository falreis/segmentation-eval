from __future__ import absolute_import
from __future__ import print_function
import os

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import tensorflow as tf
import keras.models as models
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Maximum, Add, Average, Concatenate, Lambda, Input, Conv2DTranspose
from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
import random
np.random.seed(7) # for reproducibility

import hed_constants as hc

def side_branch(classes, x, factor, morf=False):
    #kernel_morf = (11, 11)
    kernel_size = (2*factor, 2*factor)

    x = Convolution2D(classes, (1, 1), activation=None, padding='same')(x)
    x = Conv2DTranspose(classes, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    '''
    if(morf):
        #closing
        x = tf.nn.erosion2d(value=x, kernel=kernel_morf, strides=factor, padding='same', name='erosion_closing')
        x = tf.nn.dilation2d(input=x, filter=kernel_morf,strides=factor, padding='same', name='dilat_closing')

        #opening
        x = tf.nn.dilation2d(input=x, filter=kernel_morf,strides=factor, padding='same', name='dilat_opening')
        x = tf.nn.erosion2d(value=x, kernel=kernel_morf, strides=factor, padding='same', name='erosion_opening')
    '''
    return x

###########
### HED ###
###########
def model_slo(merge_name):
    inputs = Input((3, hc.height, hc.width))

    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    b1= side_branch(classes=hc.n_classes, x=x, factor=1, morf=False) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 240 240 64

    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    b2= side_branch(classes=hc.n_classes, x=x, factor=2, morf=False) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128

    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    b3= side_branch(classes=hc.n_classes, x=x, factor=4, morf=False) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256

    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    b4= side_branch(classes=hc.n_classes, x=x, factor=8, morf=False) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512

    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
    b5= side_branch(classes=hc.n_classes, x=x, factor=16, morf=False) # 480 480 1

    if(merge_name == 'avg'):
        print('avg')
        fuse = Average()([b1, b2, b3, b4, b5])

    elif(merge_name == 'add' or merge_name == 'sum'):
        print('add')
        merge_name = 'add'
        fuse = Add()([b1, b2, b3, b4, b5])

    elif(merge_name == 'max'):
        print('max')
        fuse = Maximum()([b1, b2, b3, b4, b5])

    elif(merge_name == 'maj'):
        print('maj')
        fuse = Add()([b1, b2, b3, b4, b5]) #vote procedure is defined at loss function
        #fuse = Lambda(Vote, arguments={'vote_value': vote_value})(fuse)

    fuse = Convolution2D(hc.n_classes, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

    #reshape
    ofuse = Reshape((hc.n_classes, hc.data_shape), input_shape=(hc.n_classes, hc.height, hc.width))(fuse)
    ofuse = Permute((2, 1), name='ofuse')(ofuse)

    model = Model(inputs=inputs, outputs=ofuse)
    #print(model.summary())

    return model


############
### FULL ###
############
def model_alo(merge_name):
    inputs = Input((3, hc.height, hc.width))

    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    b11 = side_branch(hc.n_classes, x, 1) # 480 480 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    b12= side_branch(hc.n_classes, x, 1) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 240 240 64

    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    b21 = side_branch(hc.n_classes, x, 2) # 480 480 1
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    b22 = side_branch(hc.n_classes, x, 2) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128

    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    b31 = side_branch(hc.n_classes, x, 4) # 480 480 1
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    b32 = side_branch(hc.n_classes, x, 4) # 480 480 1
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    b33 = side_branch(hc.n_classes, x, 4) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256

    # Block 4
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    b41= side_branch(hc.n_classes, x, 8) # 480 480 1
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    b42= side_branch(hc.n_classes, x, 8) # 480 480 1
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    b43= side_branch(hc.n_classes, x, 8) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512

    # Block 5
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    b51= side_branch(hc.n_classes, x, 16) # 480 480 1
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    b52= side_branch(hc.n_classes, x, 16) # 480 480 1
    x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
    b53= side_branch(hc.n_classes, x, 16) # 480 480 1

    if(merge_name == 'avg'):
        print('avg')
        fuse = Average()([b11, b12, b21, b22, b31, b32, b33, b41, b42, b43, b51, b52, b53])

    elif(merge_name == 'add' or merge_name == 'sum'):
        print('add')
        merge_name = 'add'
        fuse = Add()([b11, b12, b21, b22, b31, b32, b33, b41, b42, b43, b51, b52, b53])

    elif(merge_name == 'max'):
        print('max')
        fuse = Maximum()([b11, b12, b21, b22, b31, b32, b33, b41, b42, b43, b51, b52, b53])

    fuse = Convolution2D(hc.n_classes, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

    #reshape
    ofuse = Reshape((hc.n_classes, hc.data_shape), input_shape=(hc.n_classes, hc.height, hc.width))(fuse)
    ofuse = Permute((2, 1))(ofuse)

    model = Model(inputs=inputs, outputs=ofuse)
    #print(model.summary())

    return model
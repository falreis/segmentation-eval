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
from keras.layers import Maximum, Add, Average, Concatenate, Lambda, Input, Conv2DTranspose
from keras import backend as K
K.set_image_data_format('channels_first')

import numpy as np
import json
import random
np.random.seed(7) # 0bserver07 for reproducibility

#def Vote(x, vote_value):
#    vote_value -=0.5
#    return K.hard_sigmoid((x-vote_value)*5)

#import contants
import hed_constants as hc
height = hc.height
width = hc.width
data_shape =  hc.data_shape
n_classes = hc.n_classes

#import parser
import hed_args_model as ha

def side_branch(classes, x, factor):
    x = Convolution2D(classes, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(classes, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x

###########
# NETWORK #
inputs = Input((3, height, width))

# Block 1
x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
b1= side_branch(n_classes, x, 1) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 240 240 64

# Block 2
x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
b2= side_branch(n_classes, x, 2) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128

# Block 3
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
b3= side_branch(n_classes, x, 4) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256

# Block 4
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
b4= side_branch(n_classes, x, 8) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512

# Block 5
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
b5= side_branch(n_classes, x, 16) # 480 480 1

if(ha.merge_name == 'avg'):
    print('avg')
    fuse = Average()([b1, b2, b3, b4, b5])

elif(ha.merge_name == 'add' or ha.merge_name == 'sum'):
    print('add')
    merge_name = 'add'
    fuse = Add()([b1, b2, b3, b4, b5])

elif(ha.merge_name == 'max'):
    print('max')
    fuse = Maximum()([b1, b2, b3, b4, b5])

elif(ha.merge_name == 'maj'):
    print('maj')
    fuse = Add()([b1, b2, b3, b4, b5]) #vote procedure is defined at loss function
    #fuse = Lambda(Vote, arguments={'vote_value': ha.vote_value})(fuse)

elif(ha.merge_name == 'out'):
    if(ha.out_value == 1):
        fuse = b1
    elif(ha.out_value == 2):
        fuse = b2
    elif(ha.out_value == 3):
        fuse = b3
    elif(ha.out_value == 4):
        fuse = b4
    elif(ha.out_value == 5):
        fuse = b5
    else:
        print('Output not found!!')
        quit()
#endif

fuse = Convolution2D(n_classes, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

#reshape
ofuse = Reshape((n_classes, data_shape), input_shape=(n_classes, height, width))(fuse)
ofuse = Permute((2, 1), name='ofuse')(ofuse)

model = Model(inputs=inputs, outputs=ofuse)
#print(model.summary())

# Save model to JSON
if(ha.vote_value > 0):
    json_name = '../model-json/hed_kitti_model_{}_{}.json'.format(ha.merge_name, ha.vote_value)
elif(ha.out_value > 0):
    json_name = '../model-json/hed_kitti_model_{}_{}.json'.format(ha.merge_name, ha.out_value)
else:
    json_name = '../model-json/hed_kitti_model_{}.json'.format(ha.merge_name)

print(json_name)
with open(json_name, 'w') as outfile:
    outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

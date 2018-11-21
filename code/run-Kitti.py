from __future__ import absolute_import
from __future__ import print_function
import os
import datetime

#os.environ['KERAS_BACKEND'] = 'theano'
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
import argparse
np.random.seed(7) # 0bserver07 for reproducibility

print(datetime.datetime.now())

#import contants
import hed_constants as hc
height = hc.height
width = hc.width
data_shape =  hc.data_shape
n_classes = hc.n_classes

#parser
parser = argparse.ArgumentParser()
parser.add_argument("--net", type = str)
args = parser.parse_args()
net_parse = args.net
print(net_parse)

if(net_parse != None):
    #parameters
    nb_epoch = 10
    batch_size = 4

    # load the data
    train_data = np.load('./data/Kitti/train_data.npy')
    train_label = np.load('./data/Kitti/train_label.npy')

    # define parameters
    json_model, weights_file = "", ""
    if net_parse == "hed":
        json_model = 'model-json/hed_kitti_model.json'
        weights_file = "hed_kitti_weights.best.hdf5"

    elif net_parse == "rcf":
        json_model = 'model-json/rcf_kitti_model.json'
        weights_file = "rcf_kitti_weights.best.hdf5"
    #endif

    #load model
    with open(json_model) as model_file:
        net_basic = models.model_from_json(model_file.read())

    net_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    # checkpoint
    #checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    # Fit the model
    history = net_basic.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, shuffle=True, validation_split=0.20)

    # This save the trained model weights to this file with number of epochs
    if net_parse == "hed":
        net_basic.save_weights('weights/hed_model_kitti_{}.hdf5'.format(nb_epoch))
    elif net_parse == "rcf":
        net_basic.save_weights('weights/rcf_model_kitti_{}.hdf5'.format(nb_epoch))

    print(datetime.datetime.now())

else:
    print('Escolha a rede para execução >> "python run-kitti.py --net=hed"')
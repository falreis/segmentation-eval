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
parser.add_argument("--merge", type = str)
parser.add_argument("--vote", nargs='?', type= int)
args = parser.parse_args()
net_parse = args.net
merge_name = args.merge

if(args.vote):
    vote_value = args.vote
    print('Vote value: ', vote_value)
else:
    if(merge_name == "maj"):
        print('Maj operation must contain "--vote" parameter!')
        quit()

print('Neural net: ', net_parse)
print('Merge method: ', merge_name)

#se merge não for definido
if(merge_name == None):
    print('Usage >> "python run-kitti.py --net={hed,rcf} --merge={sum,avg,max,maj} --vote{0-5}"')
    quit()

if(net_parse != None):
    #parameters
    nb_epoch = 100
    batch_size = 10

    # load the data
    train_data = np.load('../data/Kitti/train_data.npy')
    train_label = np.load('../data/Kitti/train_label.npy')

    # define parameters
    json_model, weights_file = "", ""
    if(args.vote):
        json_model = '../model-json/' + net_parse + '_kitti_model_{}_{}.json'.format(merge_name, vote_value)
    else:
        json_model = '../model-json/' + net_parse + '_kitti_model_{}.json'.format(merge_name)
    #endif

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
    if(args.vote):
        hdf5_save = '../weights/' + net_parse + 'model_kitti_{}_{}_{}.hdf5'.format(merge_name, vote_value, nb_epoch)
    else:
        hdf5_save = '../weights/' + net_parse + 'model_kitti_{}_{}.hdf5'.format(merge_name, nb_epoch)

    print(datetime.datetime.now())

else:
    print('Usage >> "python run-kitti.py --net=hed --merge={sum,avg,max,maj}"')
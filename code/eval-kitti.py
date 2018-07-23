from __future__ import absolute_import
from __future__ import print_function
import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras import backend as K
K.set_image_data_format('channels_first')

import cv2
import numpy as np
import json
import visualize as vis
from skimage import io

np.random.seed(7)

# load the model:

with open('segNet_kitti_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())

# load weights
segnet_basic.load_weights("kitti_weights.best.hdf5")

# Compile model (required to make predictions)
segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

#load test data
test_data = np.load('./data/Kitti/test_data.npy')
test_label = np.load('./data/Kitti/test_label.npy')

batch_size = 1

# estimate accuracy on whole dataset using loaded weights

label_colours = np.array([[255, 0, 0], [255, 0, 255], [0, 0, 0]])

#export data
data_path = './export/Kitti/'

index = 0
for test_image in test_data:
    index += 1
    output = segnet_basic.predict_proba(test_image[np.newaxis, :])
    pred_image = vis.visualize(np.argmax(output[0],axis=1).reshape((120,408)), label_colours, False)

    image_name = data_path + str(index) + '.png' 
    io.imsave(image_name, pred_image)

print('Done')
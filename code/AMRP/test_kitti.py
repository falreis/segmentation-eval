#usage 
#python eval-kitti.py --net=segnet

#rename result files with commands below:
#rename 's/umm_/umm_road_/' 

from __future__ import absolute_import
from __future__ import print_function
import os

#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=None'

import tensorflow as tf
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_data_format('channels_first')

import cv2
import sys
import numpy as np
import json
from skimage import io
import glob
from skimage import data, transform
import argparse

np.random.seed(7)

#import contants
import hed_constants as hc
height = hc.height
width = hc.width
data_shape =  hc.data_shape
n_classes = hc.n_classes

sys.path.append("..")
import visualize as vis
from helper import *

def test(model, net, merge_name=None, set_name='test', mark=False, learn_rate=0.001, folder=None, morf=True):
    if(net != None):
        # define weights file
        if(folder != None and folder != ''):
            weights_file = '../weights/{}/{}_kitti_weight_{}.best.hdf5'.format(folder, net, merge_name)
        else:
            weights_file = '../weights/{}_kitti_weight_{}.best.hdf5'.format(net, merge_name)

        #verify mathematical morfology
        if(mark and morf):
            print('Mathematical morfology can be only applied with option --mark=False')

        print(weights_file)

        # load weights
        model.load_weights(weights_file)

        # Compile model (required to make predictions)
        sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.95, nesterov=False)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        batch_size = 16

        # estimate accuracy on whole dataset using loaded weights
        label_colours = np.array([[255, 0, 0],[255, 0, 255]])

        #load test data
        DataPath = '../datasets/Kitti/data_road/'

        #export data
        save_path = '../export/Kitti/' + net + '/' + set_name + '/'
        images_path = DataPath + set_name + 'ing/image_2/*.*'
        
        images_paths = glob.glob(images_path + "jpg") + glob.glob(images_path + "png")
        images_paths.sort()

        print(save_path)
        print(images_path)

        index = 0
        len_data = len(images_paths)
        #for test_image, image_path in zip(data, images_paths):
        for image_path in images_paths[:1]:

            #read original image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_width = original_image.shape[1]
            original_height = original_image.shape[0]

            reduced_image = cv2.resize(cv2.imread(image_path), dsize=(width, height, 3)[:2], interpolation=cv2.INTER_CUBIC)
            test_image = np.rollaxis(normalized(reduced_image), 2) 

            #predict
            output = model.predict(test_image[np.newaxis, :])
            pred_image = vis.visualize(np.argmax(output[0],axis=1).reshape((height,width)), label_colours, False)
            
            #expand predict to the size of the original image
            if(mark):
                expanded_pred = cv2.resize(pred_image, dsize=(original_width, original_height, 3)[:2], interpolation=cv2.INTER_CUBIC)
            else:
                expanded_pred = transform.resize(pred_image, (original_height, original_width, 3)).astype(np.float)

            #mark lane or create "ground-truth"
            for i in range(1, original_height):
                for j in range(1, original_width):                
                    if (expanded_pred[i, j, 2] > 0):
                        if(mark):
                            original_image[i,j,0] = 0
                            original_image[i,j,2] = 0
                        else:
                            #expanded_pred[i,j,:] = [1., -1., 1.]
                            expanded_pred[i,j,:] = [1., 0., 1.]
            
            #apply mathematical morphology
            if((not mark) and morf):
                for i in range(3, 8, 1):
                    kernel = np.ones((2*i-1,2*i-1),np.uint8)
                    expanded_pred = cv2.morphologyEx(expanded_pred, cv2.MORPH_OPEN, kernel)
                
                '''
                for i in range(3, 5, 1):
                    kernel = np.ones((2*i-1,2*i-1),np.uint8)
                    expanded_pred = cv2.morphologyEx(expanded_pred, cv2.MORPH_CLOSE, kernel)
                '''
                

            #save data
            pos = image_path.rfind('/')
            name_file = image_path[pos+1:]

            if(mark):
                io.imsave((save_path + name_file), original_image)
            else:
                io.imsave((save_path + name_file), expanded_pred)
            
            #verbose
            index += 1
            print(index, '/', len_data, end='')
            print('\r', end='')

        print("Done")
    else:
        print('Usage >> "python eval-kitti_hed.py --net=hed --merge={sum,avg,max}"')
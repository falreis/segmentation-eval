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
import constants as const
height = const.height
width = const.width
data_shape =  const.data_shape
n_classes = const.n_classes

sys.path.append("..")
import visualize as vis
from helper import *

def imFill(color_image):
    color_image = color_image.astype(np.float32)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Copy the thresholded image.
    im_floodfill = gray.copy() 
                
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = gray.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = np.bitwise_or(gray.astype(int), im_floodfill_inv.astype(int))

    return im_out


def test(model, net, merge_name=None, set_name='test', mark=False, learn_rate=0.001, folder=None, morf=True, gray=False):
    if(net != None):
        # define weights file
        if(folder != None and folder != ''):
            weights_file = '../weights/bsds/{}/{}_bsds_weight_{}.best.hdf5'.format(folder, net, merge_name)
        else:
            weights_file = '../weights/bsds/{}_bsds_weight_{}.best.hdf5'.format(net, merge_name)

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
        if(gray):
            label_colours = np.array([[0, 0, 0],[1, 1, 1]])
        else:
            label_colours = np.array([[0, 0, 0],[255, 255, 255]])

        #load test data
        DataPath = '../datasets/HED-BSDS/'

        #export data
        if(folder != None):
            save_path = '../export/HED-BSDS/{}/{}/{}/{}/'.format(folder, net, merge_name, set_name)
        else:
            save_path = '../export/HED-BSDS/{}/{}/{}/'.format(net, merge_name, set_name)

        #create folder, if not exist
        if not os.path.exists(save_path):
            print('Create path: ', save_path)
            os.makedirs(save_path)

        #define images path
        if(set_name == 'train'):
            images_path = DataPath + 'train/aug_data/0.0_1_0/*.*'
        elif(set_name == 'test'):
            images_path = DataPath + 'test/*.*'

        images_paths = glob.glob(images_path + "jpg") + glob.glob(images_path + "png")
        images_paths.sort()

        print(save_path)
        print(images_path)

        index = 0
        len_data = len(images_paths)
        #for test_image, image_path in zip(data, images_paths):
        for image_path in images_paths[:]:

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
                            if(gray):
                                expanded_pred[i,j,:] = [1, 1, 1]
                            else:
                                expanded_pred[i,j,:] = [1., 0., 1.]
            
            #apply mathematical morphology
            if((not mark) and morf):
                if(gray):
                    expanded_pred = cv2.cvtColor(expanded_pred.astype(np.float32), cv2.COLOR_BGR2GRAY)

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
        print('Read README file to learn how to use!')
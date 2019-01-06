from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools
import imutils

import sys
import os
import argparse
import glob
from skimage import io

#import constants
import constants as const  

sys.path.append("..")
from helper import *

reduced_image_size = (const.width, const.height, 3)

def load_data(mode, data_path):
    data = []
    label = []

    if mode == "train":
        filepath = data_path + '/train_pair.lst'
        datafile = open(filepath, 'r')
        lines = datafile.readlines() 
        datafile.close()

        index = 0
        len_data = len(lines)

        num_lines = 960 #9600 #28800

        if len_data > 0:
            for line in lines[0:num_lines]:
                image_file, ground_file = line.replace('\n','').split(' ')

                image_file = data_path + image_file
                ground_file = data_path + ground_file

                image = cv2.imread(image_file)
                ground = cv2.imread(ground_file)

                if(image.shape[0] > image.shape[1]):
                    image = imutils.rotate_bound(image, 90)
                    ground = imutils.rotate_bound(ground, 90)

                reduced_image = cv2.resize(image, dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
                reduced_ground = cv2.resize(ground, dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)

                data.append(np.rollaxis(normalized(reduced_image), 2))
                label.append(one_hot_kitti(reduced_ground, height = const.height, width = const.width, classes = const.n_classes))

                index += 1
                print(index, '/', len_data, end='')
                print('\r', end='')
        else:
            print('Alguma coisa errada: número de ground-truths não é igual ao de imagens.')
            quit()

    '''
    elif mode == "test":
        path = data_path + '/train_pair.lst'

        for image in images:
            reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
            data.append(np.rollaxis(normalized(reduced_image), 2))
    '''
    return np.array(data), np.array(label)

def npy(set_name='train'):
    data_path = '../datasets/HED-BSDS/'
    output_path = '../data/HED-BSDS/{}{}'

    data, label = load_data(mode=set_name, data_path=data_path)
    len_data = len(data)

    if set_name == "train":
        label = np.reshape(label,(len_data, const.data_shape, const.n_classes))
        np.save(output_path.format(set_name,"_label"), label)

    np.save(output_path.format(set_name,"_data"), data)

    print('Done')
## usage (to train set)
#python kitti_data_loader.py --set=train

from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

import sys
import os
import argparse
import glob
from skimage import io

#import constants
import hed_constants as hc  

sys.path.append("..")
from helper import *

reduced_image_size = (hc.width, hc.height, 3)

def load_data(mode, data_path):
    path = data_path + mode + "ing/image_2/"
    pathg = data_path + mode + "ing/gt_image_2/"

    data = []
    label = []

    print(path)
    images = glob.glob(path + "*.jpg") + glob.glob(path + "*.png")
    images.sort()

    if mode == "train":
        print(pathg)
        grounds = glob.glob(pathg + "*road*.jpg") + glob.glob(pathg + "*road*.png")
        grounds.sort()
        len_data = len(images)

        index = 0
        if(len(grounds) == len(images)):
            for image, ground in zip(images, grounds):
                reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
                reduced_ground = cv2.resize(cv2.imread(ground), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)

                data.append(np.rollaxis(normalized(reduced_image), 2))
                label.append(one_hot_kitti(reduced_ground, height = hc.height, width = hc.width, classes = hc.n_classes))

                index += 1
                print(index, '/', len_data, end='')
                print('\r', end='')
        else:
            print('Alguma coisa errada: número de ground-truths não é igual ao de imagens.')
            quit()

    elif mode == "test":
        for image in images:
            reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
            data.append(np.rollaxis(normalized(reduced_image), 2))

    return np.array(data), np.array(label)

def npy(set_name='train', augm=True):
    data_path = '../datasets/Kitti/data_road'
    output_path = '../data/Kitti/'

    if(augm):
        data_path += '_augmented/'
    else:
        data_path += '/'

    data, label = load_data(mode=set_name, data_path=data_path)
    len_data = len(data)

    if set_name == "train":
        label = np.reshape(label,(len_data, hc.data_shape, hc.n_classes))
        np.save((output_path + set_name + "_label"), label)

    np.save((output_path + set_name + "_data"), data)

    print('Done')
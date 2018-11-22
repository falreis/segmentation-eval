## usage (to train set)
#python kitti_data_loader.py --set=train

from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from .. import helper
import os
import argparse
import glob
from skimage import io

#import contants
import hed_constants as hc
height = hc.height
width = hc.width
data_shape =  hc.data_shape
n_classes = hc.n_classes

DataPath = './datasets/Kitti/data_road/'
reduced_image_size = (width, height, 3)

def load_data(mode):
    data = []
    label = []

    path = DataPath + mode + "_hed/image_2/"
    print(path)
    images = glob.glob(path + "*.jpg") + glob.glob(path + "*.png")
    images.sort()

    if mode == "train":
        pathg = DataPath + mode + "_hed/gt_image_2/"
        print(pathg)
        grounds = glob.glob(pathg + "*.jpg") + glob.glob(pathg + "*.png")
        grounds.sort()

        for image, ground in zip(images, grounds):
            reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
            reduced_ground = cv2.resize(cv2.imread(ground), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)

            data.append(np.rollaxis(normalized(reduced_image), 2))
            label.append(one_hot_kitti(reduced_ground, height = height, width = width, classes = n_classes))

    elif mode == "test":
        for image in images:
            reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
            data.append(np.rollaxis(normalized(reduced_image), 2))

    return np.array(data), np.array(label)


parser = argparse.ArgumentParser()
parser.add_argument("--set", type = str)
args = parser.parse_args()
set_name = args.set

print(set_name)

data, label = load_data(set_name)
len_data = len(data)

if set_name == "train":
    label = np.reshape(label,(len_data,data_shape, n_classes))
    np.save(("data/Kitti/" + set_name + "_label"), label)

np.save(("data/Kitti/" + set_name + "_data"), data)

print('Done')
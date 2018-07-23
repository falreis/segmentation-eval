## usage (to train set)
#python kitti_data_loader.py --set=train


from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os
import argparse
import glob

DataPath = './datasets/Kitti/'
height = 184 #~375/2
width = 613 #~1242/2
data_shape = height*width
n_classes = 2

reduced_image_size = (width, height, 3)

def load_data(mode):
    data = []
    label = []

    path = DataPath + mode + "/*."
    pathg = DataPath + mode + "_ground/*."

    images = glob.glob(path + "jpg") + glob.glob(path + "png")
    images.sort()

    grounds = glob.glob(pathg + "jpg") + glob.glob(pathg + "png")
    grounds.sort()

    for image, ground in zip(images, grounds):
        reduced_image = cv2.resize(cv2.imread(image), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)
        reduced_ground = cv2.resize(cv2.imread(ground), dsize=reduced_image_size[:2], interpolation=cv2.INTER_CUBIC)

        data.append(np.rollaxis(normalized(reduced_image), 2))
        label.append(one_hot_kitti(reduced_ground, height = height, width = width, classes = n_classes))
    return np.array(data), np.array(label)


parser = argparse.ArgumentParser()
parser.add_argument("--set", type = str)
args = parser.parse_args()
set_name = args.set

print(set_name)

data, label = load_data(set_name)
len_data = len(data)
label = np.reshape(label,(len_data,data_shape, n_classes))

np.save(("data/Kitti/" + set_name + "_data"), data)
np.save(("data/Kitti/" + set_name + "_label"), label)

print('Done')

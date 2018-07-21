## usage (to val set)
#python camvid_data_loader.py --set=val


from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os
import argparse

# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = './datasets/CamVid/'
data_shape = 360*480

def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):        
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + "/datasets" + txt[i][0][7:])),2))
        label.append(one_hot_it(cv2.imread(os.getcwd() + "/datasets" + txt[i][1][7:][:-1])[:,:,0]))
        #print('.',end='')
    return np.array(data), np.array(label)


parser = argparse.ArgumentParser()
parser.add_argument("--set", type = str  )
args = parser.parse_args()
set_name = args.set

print(set_name)

data, label = load_data(set_name)
len_data = len(data)
label = np.reshape(label,(len_data,data_shape,12))

np.save(("data/CamVid/" + set_name + "_data"), data)
np.save(("data/CamVid/" + set_name + "_label"), label)

print('Done')

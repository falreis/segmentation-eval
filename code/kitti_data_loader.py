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

DataPath = './Kitti/'
data_shape = 1242*375

def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        #print('.',end='')
    return np.array(data), np.array(label)


parser = argparse.ArgumentParser()
parser.add_argument("--set", type = str  )
args = parser.parse_args()
set_name = args.set

print(set_name)

data, label = load_data(set_name)
len_data = len(data)
label = np.reshape(label,(len_data,data_shape,2))

np.save(("data/Kitti/" + set_name + "_data"), data)
np.save(("data/Kitti/" + set_name + "_label"), label)

print('Done')

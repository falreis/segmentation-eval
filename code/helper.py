from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def one_hot_it(labels, height = 360, width = 480, classes = 12):
    x = np.zeros([height,width,classes])
    for i in range(height):
        for j in range(width):
            x[i,j,labels[i][j]]=1
    return x

def one_hot_kitti(labels, height = 375, width = 1242, classes = 3):
    x = np.zeros([height,width,classes])
    for i in range(height):
        for j in range(width):
            if(classes == 3):
                if (labels[i,j,0] == 0): #black (00 00 00)
                    x[i,j,0]=1
                elif (labels[i,j,2] == 0): #red (ff 00 00)
                    x[i,j,1]=1
                else: #fushia (ff 00 ff)
                    x[i,j,2]=1
            else:
                if (labels[i,j,0] == 0 or labels[i,j,2] == 0): #black (00 00 00$
                    x[i,j,0]=1
                else: #fushia (ff 00 ff)
                    x[i,j,1]=1

    return x

def one_hot_bsds(labels, height = 321, width = 481, classes = 2):
    x = np.zeros([height,width,classes])
    for i in range(height):
        for j in range(width):
            if(labels[i,j,0] != 0):
                x[i,j,1]=1
            else:
                x[i,j,0]=1
    return x
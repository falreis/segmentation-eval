from __future__ import absolute_import
from __future__ import print_function
import os

from keras.applications.vgg16 import VGG16
import model_kitti as mk
from keras.utils import plot_model
import glob
import numpy as np

#K.set_image_data_format('channels_first')
'''
print('########### VGG16 ###########')
model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#print(model.summary())
plot_model(model, to_file='_vgg16.png')
'''

print('########### ALO ###########')
model2 = mk.model_alo(merge_name='avg', side_output=False)
print(model2.summary())
#plot_model(model2, to_file='_alo.png')

'''
data_glob = glob.glob('../data/Kitti/train_data*.npy') 
label_glob = glob.glob('../data/Kitti/train_label*.npy')

augm_str = '_augm'
is_first = True

if(len(data_glob) == len(label_glob)):
    for data, label in zip(data_glob, label_glob):
        td = np.load(data)
        tl = np.load(label)

        if(is_first):
            is_first = False
            train_data = td
            train_label = tl
            print(train_data.shape)
        else:
            train_data = np.append(train_data, td, axis=0)
            train_label = np.append(train_label, tl, axis=0)
            print(train_data.shape)
'''
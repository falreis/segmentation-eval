from __future__ import absolute_import
from __future__ import print_function
import os

from keras.applications.vgg16 import VGG16
import model_kitti as mk
from keras.utils import plot_model

#K.set_image_data_format('channels_first')
print('########### VGG16 ###########')
model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#print(model.summary())
plot_model(model, to_file='_vgg16.png')

print('########### ALO ###########')
model2 = mk.model_alo(merge_name='avg', side_output=False)
#print(model2.summary())
plot_model(model2, to_file='_alo.png')

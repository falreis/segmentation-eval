#code from Htiango/Image-Segmentation
#https://github.com/Htiango/Image-Segmentation/blob/master/main/process_ground_truth.py

import scipy.io
import numpy as np

def get_first_groundTruth(path):
    """
    return first groundtruth
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)
    groundTruth = mat.get('groundTruth')
    boundary = groundTruth[0][1]['Boundaries'][0][0]
    
    #reshape
    height = boundary.shape[0]
    width = boundary.shape[1]
    boundary = boundary.reshape(height, width, 1)
    
    boundary[(boundary > 0)] = 255
    
    true_boundary = np.zeros((height, width, 3),dtype=np.uint8)
    for i in range(0,3):
        true_boundary[:,:,i] = boundary[:,:,0]
    
    return true_boundary


def get_groundTruth(path):
    """
    return the nparray of boundary (0 for boundary and 255 for area)
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)
    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size

    for i in range(label_num):
        boundary = groundTruth[0][i]['Boundaries'][0][0]
        if i == 0:
            trueBoundary = boundary
        else:
            trueBoundary += boundary

    height = trueBoundary.shape[0]
    width = trueBoundary.shape[1]
    trueBoundary = trueBoundary.reshape(height, width, 1)

    trueBoundary = 255 * np.ones([height, width, 1], dtype="uint8") - (trueBoundary > 0) * 255

    return trueBoundary


# get_groundTruth('../data/groundTruth/train/2092.mat')
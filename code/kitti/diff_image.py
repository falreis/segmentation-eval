import cv2
import numpy as np
from skimage import io
from skimage.measure import compare_ssim
import imutils
from PIL import Image

def replace_colors(image):
    data = np.array(image)

    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    mask = (red == 255) & (green == 0) & (blue == 0)
    data[:,:,:3][mask] = [0, 0, 0]

    mask = (red == 255) & (green == 0) & (blue == 255)
    data[:,:,:3][mask] = [255, 255, 255]
    return data


image1 = cv2.imread('../export/Kitti/20190102/alo/avg/test/um_road_000000.png')
image2 = cv2.imread('../export/Kitti/20190102/alo/avg/test_morf/um_road_000000.png')
thresh = image1 - image2

'''
imageA = replace_colors(image1)
imageB = replace_colors(image2)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
'''

io.imsave('../export/Kitti/20190102/alo/avg/diff_morf.png', thresh)

print('Done')
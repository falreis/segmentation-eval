import numpy as np
from PIL import Image
import glob

r1, g1, b1 = 0, 0, 0 # Original value
r2, g2, b2 = 255, 0, 0 # Value that we want to replace it with

pathg = 'gt_image_2/'
grounds = glob.glob(pathg + "*road*.jpg") + glob.glob(pathg + "*road*.png")
grounds.sort()

for ground in grounds:
    im = Image.open(ground)
    data = np.array(im)

    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:3][mask] = [r2, g2, b2]

    im = Image.fromarray(data)
    im.save(ground)

print('Done')

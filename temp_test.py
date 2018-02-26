import gabor
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from skimage.feature import hog
import numpy as np
import skimage
from skimage import io

# image = cv2.imread("data/experiment9/1.jpg")
image = io.imread("data/experiment9/1.jpg")
# 长
print(image.shape[0])
# 宽
print(image.shape[1])

print(image.shape[2])

# image = cv2.resize(image, (64, 64))
### gabor 特征
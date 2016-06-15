import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('frame00363.jpeg', 0)
imgR = cv2.imread('frame00364.jpeg', 0)

stereo = cv2.StereoBM(1, 16, 15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
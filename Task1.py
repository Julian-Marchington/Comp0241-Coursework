import numpy as np
import copy
import cv2
import matplotlib
from matplotlib import pyplot as plt
import random
import copy
from PIL import Image

# Attempting to retrieve binary mask from image
image = cv2.imread('EXAMPLE.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
cv2.imshow('Binary Image', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()
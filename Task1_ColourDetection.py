import numpy as np
import copy
import cv2
import matplotlib
from matplotlib import pyplot as plt
import random
import copy
from PIL import Image

# Attempting to retrieve binary mask from image
image = cv2.imread('/Users/julianmarchington/Desktop/Comp0241_Photos/RealPhotos/IMG_0444.JPG')

def mask_using_colour(image, lower_colour, upper_colour):

    modified_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(modified_image, lower_colour, upper_colour)

    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

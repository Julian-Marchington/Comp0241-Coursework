import numpy as np
import copy
import cv2
import matplotlib
from matplotlib import pyplot as plt
import random
import copy
from PIL import Image
from collections import deque

# Attempting to retrieve binary mask from image
image = cv2.imread('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000030.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define blue range
lower_blue = np.array([100, 50, 50]) 
upper_blue = np.array([140, 255, 255])

lower_clouds = np.array([85, 5, 180]) 
upper_clouds = np.array([135, 80, 255])

lower_test = np.array([60, 0, 0])
upper_test = np.array([179, 173, 134])

lower_bound = np.array([69, 20, 0])
upper_bound = np.array([179, 255, 255])

# Create masks
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
test_mask = cv2.inRange(hsv, lower_test, upper_test)
another_mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, white_mask)
combined_mask = cv2.bitwise_or(combined_mask, test_mask)
combined_mask = cv2.bitwise_or(combined_mask, another_mask)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)    # Remove small noise

# Create the final binary mask
final_mask = cv2.bitwise_and(image, image, mask=cleaned_mask)

# Show the segmented AO
cv2.imshow('Segmented AO', cleaned_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
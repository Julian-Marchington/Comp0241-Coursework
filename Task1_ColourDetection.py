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
image = cv2.imread('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000046.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define blue range
lower_blue = np.array([100, 50, 50])  # Adjust for your image
upper_blue = np.array([140, 255, 255])

# Define white range
lower_white = np.array([90, 10, 180])  # Light blue/white-ish range
upper_white = np.array([130, 60, 255])

lower_clouds = np.array([85, 5, 180])  # Lower Saturation for faint clouds
upper_clouds = np.array([135, 80, 255])  # Higher Value to capture bright clouds

# Green range for vegetation
lower_green = np.array([30, 40, 40])  # Greenish tones
upper_green = np.array([90, 255, 255])

# Brown/tan range for land
lower_brown = np.array([10, 40, 40])  # Earthy tones
upper_brown = np.array([30, 255, 200])  # Adjust upper limit if needed

# Create masks
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
#green_mask = cv2.inRange(hsv, lower_green, upper_green)
#brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, white_mask)
#combined_mask = cv2.bitwise_or(combined_mask, green_mask)
#combined_mask = cv2.bitwise_or(combined_mask, brown_mask)

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
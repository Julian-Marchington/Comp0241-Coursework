import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Path to the directory containing images
image_directory = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/'  # Replace with your image directory

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png') or f.endswith('.jpg')]

def colourDetection(image):

    # Define color ranges for segmentation
    lower_blue = np.array([100, 50, 50]) 
    upper_blue = np.array([140, 255, 255])
    lower_clouds = np.array([85, 5, 180]) 
    upper_clouds = np.array([135, 80, 255])
    lower_test = np.array([60, 0, 0])
    upper_test = np.array([179, 173, 134])
    lower_bound = np.array([69, 20, 0])
    upper_bound = np.array([179, 255, 255])

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for the defined color ranges
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
    test_mask = cv2.inRange(hsv, lower_test, upper_test)
    another_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Combine the masks
    combined_mask = cv2.bitwise_or(blue_mask, white_mask)
    combined_mask = cv2.bitwise_or(combined_mask, test_mask)
    combined_mask = cv2.bitwise_or(combined_mask, another_mask)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    return cleaned_mask


def circleDetection(image):

    # Apply Gaussian Blur to reduce noise on the cleaned mask directly
    blurred_mask = cv2.GaussianBlur(image, (9, 9), 2)

    # Use Hough Circle Transform to detect circles in the blurred mask
    circles = cv2.HoughCircles(blurred_mask, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1, 
                                minDist=20, 
                                param1=50, 
                                param2=30, 
                                minRadius=0, 
                                maxRadius=0)

    # Create a mask for the detected circles
    circle_mask = np.zeros_like(blurred_mask)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # Only takes the first detected circle
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness =-1)
    
    return circle_mask

def combineMasks(cleaned_mask, circle_mask, image):

    # Combine the color mask and the circle mask
    final_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=circle_mask)

    # Create the output image by applying the final mask
    output_image = cv2.bitwise_and(image, image, mask=final_mask)

    return output_image
import numpy as np
import cv2

def colourDetection(image):
    """
    Segments the image based on predefined HSV color ranges.
    """
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

def circleDetection(image, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
    """
    Detects circles in the input image using Hough Circle Transform.
    """
    # Apply Gaussian Blur to reduce noise
    blurred_mask = cv2.GaussianBlur(image, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # Create a mask for the detected circles
    circle_mask = np.zeros_like(blurred_mask)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0]:
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, 255, thickness=-1)

    return circle_mask

def combineMasks(cleaned_mask, circle_mask):
    """
    Combines color-based and circle-based masks to create a final segmentation.
    """
    # Combine the color mask and the circle mask
    final_mask = cv2.bitwise_and(cleaned_mask, circle_mask)
    return final_mask

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Path to the directory containing images
image_directory = 'C:/Users/nehit/OneDrive/Pictures/Camera Roll/'  # Replace with your image directory
image_filenames = ['1.jpg', '2.jpg']  # Add your image filenames here

# Define color ranges for segmentation
lower_blue = np.array([100, 50, 50]) 
upper_blue = np.array([140, 255, 255])

lower_clouds = np.array([85, 5, 180]) 
upper_clouds = np.array([135, 80, 255])

lower_test = np.array([60, 0, 0])
upper_test = np.array([179, 173, 134])

lower_bound = np.array([69, 20, 0])
upper_bound = np.array([179, 255, 255])

# Function to extract centroid from an image
def extract_centroid(image_path):
    # Load the input image
    image = cv2.imread(image_path)

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

    # Apply Gaussian Blur to reduce noise on the cleaned mask directly
    blurred_mask = cv2.GaussianBlur(cleaned_mask, (9, 9), 2)

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
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)

    # Combine the color mask and the circle mask
    final_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=circle_mask)

    # Find contours in the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the moments of the largest contour
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:  # Avoid division by zero
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            return (centroid_x, centroid_y)
    
    return None  # Return None if no centroid is found

# List to store centroids
centroids = []

# Process each image and extract centroids
for image_filename in image_filenames:
    image_path = os.path.join(image_directory, image_filename)
    centroid = extract_centroid(image_path)
    if centroid is not None:
        centroids.append(centroid)

centroids_array = np.array(centroids)

# Plotting the centroid movement
plt.figure(figsize=(10, 5))
plt.plot(centroids_array[:, 0], centroids_array[:, 1], label='Original Centroid Path')
plt.title('Centroid Movement Over Time')
plt.xlabel('X Coordinate (pixels)')
plt.ylabel('Y Coordinate (pixels)')
plt.legend()
plt.grid()
plt.show()

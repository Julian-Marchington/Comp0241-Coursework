import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

# Path to the directory containing images
image_path_1 = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/Image.png'  # Replace with your image directory
image_path_2 = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/Image_2.png'  # Replace with your image directory

def get_centroid(image_path):

    # Define color ranges for segmentation
    lower_blue = np.array([100, 50, 50]) 
    upper_blue = np.array([140, 255, 255])

    lower_clouds = np.array([85, 5, 180]) 
    upper_clouds = np.array([135, 80, 255])

    lower_test = np.array([60, 0, 0])
    upper_test = np.array([179, 173, 134])

    lower_bound = np.array([69, 20, 0])
    upper_bound = np.array([179, 255, 255])

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
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness =-1)

    # Combine the color mask and the circle mask
    final_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=circle_mask)

    # Create the output image by applying the final mask
    output_image = cv2.bitwise_and(image, image, mask=final_mask)

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
            
            # Highlight the centroid on the output image
            cv2.circle(output_image, (centroid_x, centroid_y), 10, (255, 0, 255), -1)
            print(f"Centroid of AOI: ({centroid_x}, {centroid_y})")
        else:
            print("No valid centroid found (area is zero).")
    else:
        print("No contours found in the final mask.")
    
    return output_image, centroid_x, centroid_y

pos1_image, pos1_centroid_x, pos1_centroid_y = get_centroid(image_path_1)
pos2_image, pos2_centroid_x, pos2_centroid_y = get_centroid(image_path_2)

print("Shape of Image 1:", pos1_image.shape)
print("Shape of Image 2:", pos2_image.shape)

# Display the final output image with centroid marked
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Final Output with Centroid for Image 1')
plt.imshow(cv2.cvtColor(pos1_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Final Output with Centroid for Image 2')
plt.imshow(cv2.cvtColor(pos2_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Example centroids for two images (replace with your detected centroids)
centroid1 = (pos1_centroid_x, pos1_centroid_y)  # From image 1
centroid2 = (pos2_centroid_x, pos2_centroid_y)  # From image 2

# Compute the vector components between the centroids
dx = centroid2[0] - centroid1[0]
dy = centroid2[1] - centroid1[1]

# Calculate the angle using atan2
angle_radians = math.atan2(dy, dx)  # Angle in radians
angle_degrees = math.degrees(angle_radians)  # Convert to degrees

# Ensure the angle is within [0, 360] degrees
if angle_degrees < 0:
    angle_degrees += 360

# Print the results
print(f"Centroid 1: {centroid1}")
print(f"Centroid 2: {centroid2}")
print(f"Vector: dx = {dx}, dy = {dy}")
print(f"Angle (radians): {angle_radians:.2f}")
print(f"Angle (degrees): {angle_degrees:.2f}")
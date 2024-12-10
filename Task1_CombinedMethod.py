import numpy as np
import cv2

# Load the input image
image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000059.png'
image = cv2.imread(image_path)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define your HSV ranges
# Adjust these based on your tests
lower_blue = np.array([100, 50, 50]) 
upper_blue = np.array([140, 255, 255])

lower_clouds = np.array([85, 5, 180]) 
upper_clouds = np.array([135, 80, 255])

lower_test = np.array([60, 0, 0])
upper_test = np.array([179, 173, 134])

lower_bound = np.array([69, 20, 0])
upper_bound = np.array([179, 255, 255])

# Create color masks
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
test_mask = cv2.inRange(hsv, lower_test, upper_test)
another_mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Combine masks using OR
combined_mask = cv2.bitwise_or(blue_mask, white_mask)
combined_mask = cv2.bitwise_or(combined_mask, test_mask)
combined_mask = cv2.bitwise_or(combined_mask, another_mask)

# Morphological operations to clean the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)    # Remove noise

# -----------------------------------
# Edge/Shape Based Mask (Hough Circle)
# -----------------------------------

# Convert to grayscale and blur for Hough
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough transform
# Adjust parameters (param1, param2, minRadius, maxRadius) as needed
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=50, 
    param1=100, 
    param2=30, 
    minRadius=0, 
    maxRadius=0
)

circle_mask = np.zeros_like(cleaned_mask)  # same size as the cleaned_mask

if circles is not None:
    circles = np.uint16(np.around(circles))
    # Use the first detected circle or refine logic if multiple are found
    for i in circles[0, :1]:
        center_x, center_y, radius = i
        # Draw a filled white circle on circle_mask
        cv2.circle(circle_mask, (center_x, center_y), radius, 255, thickness=-1)

# Combine the color mask and the circle mask
final_mask = cv2.bitwise_and(cleaned_mask, circle_mask)

# Apply the final mask to the original image
result = cv2.bitwise_and(image, image, mask=final_mask)

# Display or save the results
cv2.imshow('Combined Mask', final_mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the final result
#cv2.imwrite('combined_mask_result.png', result)
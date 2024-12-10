import cv2
import numpy as np

# Load your input image
image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000069.png'
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a blur to reduce noise and improve circle detection
blurred_mask = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough transform
circles = cv2.HoughCircles(
    blurred_mask, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=20, 
    param1=50, 
    param2=30, 
    minRadius=0, 
    maxRadius=0)

# Initialize a mask the same size as the image
circle_mask = np.zeros_like(blurred_mask)

if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # Only takes the first detected circle
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness =-1)

# Now mask the image to isolate the globe
masked_globe = cv2.bitwise_and(img, img, mask=circle_mask)

# Save or display the results
cv2.imshow('Masked Globe', masked_globe)
cv2.waitKey(0)
cv2.destroyAllWindows()
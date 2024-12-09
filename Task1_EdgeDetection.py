import cv2
import numpy as np

# Load your input image
image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000079.png'
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a blur to reduce noise and improve circle detection
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Use Hough Circle Transform to find the globe
# Adjust parameters as needed:
# dp=1.2: Inverse ratio of resolution
# minDist: Minimum distance between circle centers
# param1: Upper threshold for Canny edge
# param2: Accumulator threshold for circle detection (smaller = more circles)
# minRadius, maxRadius: Adjust based on your globe size in the image
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

# Initialize a mask the same size as the image
mask = np.zeros(img.shape[:2], dtype=np.uint8)

if circles is not None:
    circles = np.uint16(np.around(circles))
    # Assuming the first detected circle is the globe
    for i in circles[0, :1]:
        center_x, center_y, radius = i
        # Draw a filled white circle on the mask
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)

# Now mask the image to isolate the globe
masked_globe = cv2.bitwise_and(img, img, mask=mask)

# Save or display the results
cv2.imwrite('masked_globe.png', masked_globe)
cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Masked Globe', masked_globe)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 
import numpy as np
  
img = cv2.imread("/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000069.png") # Read image 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert image to grayscale

# Apply GaussianBlur to reduce noise
gray_img_blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(gray_img_blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)


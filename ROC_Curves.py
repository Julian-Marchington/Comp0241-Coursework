import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Directories
image_directory = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/'  # Your input images
mask_directory = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/masks/'  # Your ground truth masks

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Define a function to generate a predicted mask using Method A (Example)
def method_a(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define ranges (found through ColourRangeDetector.py)
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

    return cleaned_mask

def method_b(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    return circle_mask

# Define a function to generate a predicted mask using Method B (combined methods)
def method_c(image):
    # Replace with your combined approach logic
    # For demonstration, let's assume you used the code above that combined all masks:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50]) 
    upper_blue = np.array([140, 255, 255])
    lower_clouds = np.array([85, 5, 180]) 
    upper_clouds = np.array([135, 80, 255])
    lower_test = np.array([60, 0, 0])
    upper_test = np.array([179, 173, 134])
    lower_bound = np.array([69, 20, 0])
    upper_bound = np.array([179, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
    test_mask = cv2.inRange(hsv, lower_test, upper_test)
    another_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    combined_mask = cv2.bitwise_or(blue_mask, white_mask)
    combined_mask = cv2.bitwise_or(combined_mask, test_mask)
    combined_mask = cv2.bitwise_or(combined_mask, another_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    blurred_mask = cv2.GaussianBlur(cleaned_mask, (9, 9), 2)

    circles = cv2.HoughCircles(blurred_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circle_mask = np.zeros_like(blurred_mask)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # take the first circle
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)

    final_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=circle_mask)
    return final_mask

# ROC and AUC computation
def evaluate_roc(predicted_mask, ground_truth_mask):

    ground_truth_mask = ground_truth_mask[:, :, 0]
    
    # Ensure masks are binary and probabilities
    y_true = (ground_truth_mask > 0).astype(int).flatten()
    y_pred = (predicted_mask / 255.0).flatten()  # Normalize if needed
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def compute_tp_fp(predicted_mask, ground_truth_mask):
    # Convert ground truth mask to binary (if needed)
    ground_truth_mask = (ground_truth_mask[:, :, 0] > 0).astype(int)
    
    # Ensure the predicted mask is binary
    predicted_mask_binary = (predicted_mask > 0).astype(int)

    # Compute True Positives and False Positives
    TP = np.sum((predicted_mask_binary == 1) & (ground_truth_mask == 1))
    FP = np.sum((predicted_mask_binary == 1) & (ground_truth_mask == 0))

    return TP, FP


# Load images and masks
images = load_images_from_folder(image_directory)
masks = load_images_from_folder(mask_directory)

# To store TP and FP values for each method
tp_fp_a, tp_fp_b, tp_fp_c = [], [], []

for i in range(len(images)):
    # Generate predicted masks using Method A, B, and C
    predicted_mask_a = method_a(images[i])
    predicted_mask_b = method_b(images[i])
    predicted_mask_c = method_c(images[i])

    # Compute TP and FP for Method A
    TP_a, FP_a = compute_tp_fp(predicted_mask_a, masks[i])
    tp_fp_a.append((TP_a, FP_a))

    # Compute TP and FP for Method B
    TP_b, FP_b = compute_tp_fp(predicted_mask_b, masks[i])
    tp_fp_b.append((TP_b, FP_b))

    # Compute TP and FP for Method C
    TP_c, FP_c = compute_tp_fp(predicted_mask_c, masks[i])
    tp_fp_c.append((TP_c, FP_c))

# Scatter plot for Method A
tp_a, fp_a = zip(*tp_fp_a)  # Unpack TP and FP values
plt.scatter(fp_a, tp_a, color='blue', label='Method A')
plt.xlabel('False Positives (FP)')
plt.ylabel('True Positives (TP)')
plt.title('TP/FP Plot for Method A')
plt.legend()
plt.show()

# Scatter plot for Method B
tp_b, fp_b = zip(*tp_fp_b)  # Unpack TP and FP values
plt.scatter(fp_b, tp_b, color='green', label='Method B')
plt.xlabel('False Positives (FP)')
plt.ylabel('True Positives (TP)')
plt.title('TP/FP Plot for Method B')
plt.legend()
plt.show()

# Scatter plot for Method C
tp_c, fp_c = zip(*tp_fp_c)  # Unpack TP and FP values
plt.scatter(fp_c, tp_c, color='red', label='Method C')
plt.xlabel('False Positives (FP)')
plt.ylabel('True Positives (TP)')
plt.title('TP/FP Plot for Method C')
plt.legend()
plt.show()
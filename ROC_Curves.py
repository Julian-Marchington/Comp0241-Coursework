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

# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, title):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Image {title}')
    plt.legend(loc='lower right')
    plt.show()

# Load images and masks
images = load_images_from_folder(image_directory)
masks = load_images_from_folder(mask_directory)

# To store FPR, TPR, and AUC for each method
fpr_list_a, tpr_list_a = [], []
fpr_list_b, tpr_list_b = [], []
fpr_list_c, tpr_list_c = [], []
roc_auc_list_a, roc_auc_list_b, roc_auc_list_c = [], [], []

for i in range(len(images)):
    # Generate predicted masks using Method A and Method B
    predicted_mask_a = method_a(images[i])
    predicted_mask_b = method_b(images[i])  
    predicted_mask_c = method_c(images[i])

    # Evaluate ROC and AUC for Method A
    fpr_a, tpr_a, roc_auc_a = evaluate_roc(predicted_mask_a, masks[i])
    fpr_list_a.append(fpr_a)
    tpr_list_a.append(tpr_a)
    roc_auc_list_a.append(roc_auc_a)

    # Evaluate ROC and AUC for Method B
    fpr_b, tpr_b, roc_auc_b = evaluate_roc(predicted_mask_b, masks[i])
    fpr_list_b.append(fpr_b)
    tpr_list_b.append(tpr_b)
    roc_auc_list_b.append(roc_auc_b)

    # Evaluate ROC and AUC for Method C
    fpr_c, tpr_c, roc_auc_c = evaluate_roc(predicted_mask_c, masks[i])
    fpr_list_c.append(fpr_c)
    tpr_list_c.append(tpr_c)

# Compute the mean TPR for Method A
mean_tpr_a = np.mean([np.interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in zip(fpr_list_a, tpr_list_a)], axis=0)
mean_fpr = np.linspace(0, 1, 100)  # Common FPR range for interpolation

# Compute the mean TPR for Method B
mean_tpr_b = np.mean([np.interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in zip(fpr_list_b, tpr_list_b)], axis=0)

# Compute the mean TPR for Method C
mean_tpr_c = np.mean([np.interp(np.linspace(0, 1, 100), fpr, tpr) for fpr, tpr in zip(fpr_list_c, tpr_list_c)], axis=0)

# Add Mean ROC Curve to Method A Plot
plt.figure()
#for i in range(len(fpr_list_a)):
#    plt.plot(fpr_list_a[i], tpr_list_a[i], alpha=0.5, label=f'Image {i} (AUC = {roc_auc_list_a[i]:.2f})')
plt.plot(mean_fpr, mean_tpr_a, color='blue', label='Mean ROC', linewidth=2)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Method A with Mean')
plt.legend(loc='lower right')
plt.show()

# Add Mean ROC Curve to Method B Plot
plt.figure()
#for i in range(len(fpr_list_b)):
#    plt.plot(fpr_list_b[i], tpr_list_b[i], alpha=0.5, label=f'Image {i} (AUC = {roc_auc_list_b[i]:.2f})')
plt.plot(mean_fpr, mean_tpr_b, color='blue', label='Mean ROC', linewidth=2)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Method B with Mean')
plt.legend(loc='lower right')
plt.show()

# Add Mean ROC Curve to Method C Plot
plt.figure()
#for i in range(len(fpr_list_c)):
#    plt.plot(fpr_list_c[i], tpr_list_c[i], alpha=0.5, label=f'Image {i} (AUC = {roc_auc_list_c[i]:.2f})')
plt.plot(mean_fpr, mean_tpr_c, color='blue', label='Mean ROC', linewidth=2)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Method C with Mean')
plt.legend(loc='lower right')
plt.show()
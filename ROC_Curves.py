import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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


def load_images_from_folder(folder, grayscale=False):
    images = []
    for filename in sorted(os.listdir(folder)):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            if grayscale:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale if needed
            else:
                img = cv2.imread(filepath)
            if img is not None:
                images.append(img)
    return images

def binarize_mask(mask, threshold=128):
    """
    Convert a mask to binary (0 or 1) based on a threshold.
    """
    _, binary_mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)  # Use 1 instead of 255
    return binary_mask

def compute_roc_curve(y_true, y_pred, method_name, ax):
    """
    Compute and plot the ROC scatter plot for a given set of ground truth and predicted masks.
    """
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)

    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
    roc_auc = auc(fpr, tpr)

    # Scatter plot the ROC points
    ax.scatter(fpr, tpr, label=f"{method_name} (AUC = {roc_auc:.2f})", alpha=0.6)

    # Plot diagonal for random chance
    diagonal = np.linspace(0, 1, 100)
    ax.plot(diagonal, diagonal, linestyle='--', color='blue', label='Random Chance')

    ax.set_title(f"{method_name} ROC Scatterplot")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid()


def main():
    # Directories
    image_directory = 'C:/Users/nehit/Downloads/temp/corrected/Dataset_v3/images'
    mask_directory = 'C:/Users/nehit/Downloads/temp/corrected/Dataset_v3/masks'

    # Check directories
    if not os.path.exists(image_directory) or not os.path.exists(mask_directory):
        print("Error: One or more directories do not exist.")
        return

    # Load images and ground truth masks
    images = load_images_from_folder(image_directory)
    masks = load_images_from_folder(mask_directory, grayscale=True)
    ground_truth = [binarize_mask(mask).flatten() for mask in masks]

    # Generate predicted masks for each method
    predicted_mask_a = [method_a(img).flatten() for img in images]
    predicted_mask_b = [method_b(img).flatten() for img in images]
    predicted_mask_c = [method_c(img).flatten() for img in images]

    # Create a figure for each method
    for method_index, (predicted_masks, method_name) in enumerate(
        [(predicted_mask_a, "Method A"), (predicted_mask_b, "Method B"), (predicted_mask_c, "Method C")]
    ):
        all_fpr = []
        all_tpr = []

        # Compute ROC points for all images for the current method
        for i in range(len(images)):
            fpr, tpr, _ = roc_curve(ground_truth[i], predicted_masks[i])
            all_fpr.extend(fpr)
            all_tpr.extend(tpr)

        # Create a scatter plot for the current method
        plt.figure(figsize=(10, 8))
        plt.scatter(all_fpr, all_tpr, alpha=0.5, label=method_name)

        # Plot diagonal for random chance
        diagonal = np.linspace(0, 1, 100)
        plt.plot(diagonal, diagonal, linestyle='--', color='blue', label='Random Chance')

        plt.title(f"ROC Scatterplot for {method_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()


import numpy as np
import cv2
import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Paths to your test images and corresponding ground truth masks
test_image_paths = sorted(glob.glob('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/*.png'))
gt_mask_paths = sorted(glob.glob('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/masks/*.png'))

# Initialize lists to store all predictions and ground truths
all_scores = []
all_ground_truths = []

for img_path, gt_path in zip(test_image_paths, gt_mask_paths):
    # Load image (if needed) and ground truth mask
    image = cv2.imread(img_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 128).astype(np.uint8)  # ensure binary (0 or 1)

    # Predict the probability map or continuous score map for the segmentation method
    # This will differ depending on your method. For example:
    # predicted_scores = model.predict(image) # Hypothetical
    # Here we assume 'predicted_scores' is an array of the same size as gt_mask
    # with values in [0, 1] indicating probability of being foreground.
    # If you only have binary predictions, consider them as predicted_scores = binary_mask.
    predicted_scores = np.random.rand(*gt_mask.shape)  # Random scores for demonstration

    # Flatten arrays
    flat_scores = predicted_scores.flatten()
    flat_gt = gt_mask.flatten()

    # Append to global lists
    all_scores.append(flat_scores)
    all_ground_truths.append(flat_gt)

# Convert lists to numpy arrays
all_scores = np.concatenate(all_scores)
all_ground_truths = np.concatenate(all_ground_truths)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(all_ground_truths, all_scores)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='Method ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Segmentation Method')
plt.legend(loc="lower right")
plt.show()
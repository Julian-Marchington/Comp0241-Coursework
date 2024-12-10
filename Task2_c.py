import cv2
import numpy as np

def estimate_height(mask_path, reference_height_meters, reference_pixels):
    """
    Estimates the AO's height above the ground.

    Parameters:
        mask_path (str): Path to the binary mask image.
        reference_height_meters (float): Real-world height of a reference object (meters).
        reference_pixels (int): Pixel height of the reference object in the image.

    Returns:
        float: Estimated AO height above the ground (meters).
    """
    # Load the binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Threshold to ensure binary mask (if needed)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours to isolate the AO
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    # Assume the largest contour is the AO
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the AO
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Lowest point of the AO is the bottom of the bounding box
    lowest_point_pixel = y + h

    # Convert pixel height to meters
    scaling_factor = reference_height_meters / reference_pixels
    ao_height_meters = lowest_point_pixel * scaling_factor

    return ao_height_meters

# Example Usage
if __name__ == "__main__":
    # Path to the binary mask image
    mask_image_path = "path_to_your_binary_mask.png"

    # Real-world height and pixel height of the reference object
    reference_height_meters = 0.5  # e.g., 50 cm
    reference_pixels = 200  # Pixel height of the reference object

    ao_height = estimate_height(mask_image_path, reference_height_meters, reference_pixels)
    print(f"Estimated AO height above the ground: {ao_height:.2f} meters")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from Task1_Functions import combineMasks, circleDetection, colourDetection

def compute_rotation_matrix(keypoints1, keypoints2):
    """
    Compute the essential matrix and derive the rotation matrix
    between two sets of keypoints.
    """
    E, _ = cv2.findEssentialMat(keypoints1, keypoints2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, _, _ = cv2.recoverPose(E, keypoints1, keypoints2)
    return R

def compute_rotation_angle(R):
    """
    Compute the rotation angle from the rotation matrix.
    """
    theta = np.arccos((np.trace(R) - 1) / 2.0)
    return np.degrees(theta)  # Convert to degrees for better interpretation

def visualize_roi(frame, mask, keypoints):
    """
    Visualize the region of interest (ROI) and detected keypoints.
    """
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(roi, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow('Region of Interest and Keypoints', roi)
    return roi

def plot_metrics(timestamps, rotations, rotation_periods):
    """
    Plot metrics like angular displacement and detected rotation periods.
    """
    plt.figure(figsize=(14, 8))

    # Plot angular displacement
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, rotations, label="Angular Displacement (degrees)", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Displacement (degrees)")
    plt.title("Angular Displacement Over Time")
    plt.legend()
    plt.grid(True)

    # Plot rotation periods
    if rotation_periods:
        plt.subplot(2, 1, 2)
        plt.hist(rotation_periods, bins=10, color='green', alpha=0.7, label="Detected Rotation Periods")
        plt.xlabel("Rotation Period (s)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Detected Rotation Periods")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def compute_weighted_average(rotation_periods):
    """
    Compute a final rotation period using a weighted average based on consistency.
    More frequent periods are given higher weights.
    """
    unique_periods, counts = np.unique(rotation_periods, return_counts=True)
    weights = counts / np.sum(counts)
    weighted_average = np.sum(unique_periods * weights)
    return weighted_average

def measure_rotation_period_with_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera (ID 0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    sift = cv2.SIFT_create()

    fps = 30  # Assume 30 FPS for the camera as an approximation
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    initial_rotation = None

    frame_count = 0
    rotations = []
    timestamps = []
    rotation_periods = []
    cumulative_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect SIFT keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if prev_frame is not None:
            # Match features between the previous frame and current frame
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(prev_descriptors, descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) > 8:
                src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                R = compute_rotation_matrix(src_pts, dst_pts)

                if initial_rotation is None:
                    initial_rotation = R

                rel_rotation = R @ np.linalg.inv(initial_rotation)
                angle = compute_rotation_angle(rel_rotation)

                # Ensure angle continuity to avoid resets
                if rotations and abs(angle - rotations[-1]) > 180:
                    if angle > rotations[-1]:
                        angle -= 360
                    else:
                        angle += 360

                rotations.append(angle)
                timestamps.append(frame_count / fps)

                # Update cumulative angle
                if len(rotations) > 1:
                    cumulative_angle += abs(rotations[-1] - rotations[-2])

                # Detect a full rotation
                if cumulative_angle >= 360:
                    time_span = timestamps[-1] - timestamps[0]
                    period = time_span * (360 / cumulative_angle)
                    if 300 <= period <= 600:  # Restrict to reasonable range (5-10 minutes)
                        rotation_periods.append(period)
                    cumulative_angle = 0  # Reset for next rotation

        # Create a binary mask for the region of interest (ROI)
        mask1 = colourDetection(frame)
        mask2 = circleDetection(mask1)
        mask = combineMasks(mask1, mask2)
        cv2.circle(mask, (gray.shape[1] // 2, gray.shape[0] // 2), min(gray.shape[:2]) // 4, 255, -1)

        # Visualize ROI and keypoints
        visualize_roi(frame, mask, keypoints)

        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot results
    plot_metrics(timestamps, rotations, rotation_periods)

    if rotation_periods:
        final_rotation_period = compute_weighted_average(rotation_periods)
        print(f"Final Rotation Period (Weighted Average): {final_rotation_period:.2f} seconds")

    return rotations, timestamps, rotation_periods, final_rotation_period if rotation_periods else None

if __name__ == "__main__":
    print("Point your camera at the rotating object.")
    rotations, timestamps, rotation_periods, final_rotation_period = measure_rotation_period_with_camera()

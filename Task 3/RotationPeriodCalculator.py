import cv2
import numpy as np
import math
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from Task1_Functions import combineMasks, circleDetection, colourDetection

def detect_and_track_features(frame, prev_keypoints, orb, bf_matcher, prev_descriptors):
    """
    Detects and tracks features across frames using ORB keypoints and descriptors.
    """
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    matches = []

    if prev_keypoints is not None and prev_descriptors is not None:
        matches = bf_matcher.knnMatch(prev_descriptors, descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        matched_points = [(prev_keypoints[m.queryIdx].pt, keypoints[m.trainIdx].pt) for m in good_matches]
        return keypoints, descriptors, matched_points

    return keypoints, descriptors, []

def mark_rotation(features_visibility, frame_count, fps):
    """
    Determines if a rotation has occurred based on the cyclic visibility of features.
    """
    for feature_id, visibility in features_visibility.items():
        if len(visibility) >= 2:
            first_visible, second_visible = visibility[-2:]
            if first_visible and not second_visible:
                # Feature disappeared, possibly completed a rotation
                rotation_period = (frame_count - visibility[0][1]) / fps
                visibility.clear()  # Reset visibility for the next cycle
                return True, rotation_period
    return False, None

def measurePeriod(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    orb = cv2.ORB_create(nfeatures=300)  # Limit to 300 features for speed
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rotation_count = 0
    frame_count = 0
    prev_keypoints, prev_descriptors = None, None
    features_visibility = {}
    rotation_timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

        # Segment the rotating object
        mask1 = colourDetection(frame)
        mask2 = circleDetection(mask1)
        mask = combineMasks(mask1, mask2)

        # Focus only on the circular region
        circular_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Detect and track features within the circular region
        keypoints, descriptors, matched_points = detect_and_track_features(
            circular_frame, prev_keypoints, orb, bf_matcher, prev_descriptors
        )

        # Update feature visibility and check for rotation
        for i, (prev_pt, curr_pt) in enumerate(matched_points):
            feature_id = f"feature_{i}"
            if feature_id not in features_visibility:
                features_visibility[feature_id] = []

            visible_now = True
            features_visibility[feature_id].append((visible_now, frame_count))

        # Detect rotation
        rotation_detected, rotation_period = mark_rotation(features_visibility, frame_count, fps)
        if rotation_detected:
            rotation_count += 1
            rotation_timestamps.append(rotation_period)
            print(f"Rotation {rotation_count}: {rotation_period:.2f} seconds")

        # Update previous keypoints and descriptors
        prev_keypoints, prev_descriptors = keypoints, descriptors

        # Visualization: Draw keypoints and update markers
        for keypoint in keypoints:
            cv2.circle(frame, (int(keypoint.pt[0]), int(keypoint.pt[1])), 3, (0, 255, 0), -1)

        # Overlay markers for frame count, rotation count, and periods
        cv2.putText(frame, f"Frames: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Rotations: {rotation_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if rotation_timestamps:
            cv2.putText(frame, f"Period: {rotation_timestamps[-1]:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display frame
        cv2.imshow('Rotating Object Visualization', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = '/home/ayush/Documents/roboticsAILabs/Comp0241-Coursework/Task 3/video.mov'
    measurePeriod(video_path)

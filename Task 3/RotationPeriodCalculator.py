import cv2
import numpy as np
import math
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Task1_Functions import combineMasks, circleDetection, colourDetection

def get_largest_roundest_contour(contours):
    """
    Finds the largest roundest contour based on circularity.
    """
    roundest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        if circularity > 0.3 and area > max_area:  # Adjust threshold as needed
            roundest_contour = contour
            max_area = area

    return roundest_contour

def measurePeriod(video_path):
    """
    Measures the rotation period of an object in a video using color and contour detection, focusing on a central ROI.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the central region of interest (ROI)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_center_x = frame_width // 2
    roi_center_y = frame_height // 2
    roi_width = frame_width // 4  # Adjust size based on the object size
    roi_height = frame_height // 4

    total_angle = 0
    frame_count = 0
    rotation_count = 0
    p0 = None
    old_gray_roi = None
    old_roi_size = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Segment the rotating object
        mask1 = colourDetection(frame)
        mask2 = circleDetection(mask1)
        mask = combineMasks(mask1, mask2)

        # Apply the central ROI to the mask
        mask = mask[roi_center_y - roi_height // 2:roi_center_y + roi_height // 2,
                    roi_center_x - roi_width // 2:roi_center_x + roi_width // 2]

        roi_frame = frame[roi_center_y - roi_height // 2:roi_center_y + roi_height // 2,
                          roi_center_x - roi_width // 2:roi_center_x + roi_width // 2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter contours by size and circularity
            filtered_contours = [
                c for c in contours if cv2.contourArea(c) > 500  # Area threshold
            ]
            roundest_contour = get_largest_roundest_contour(filtered_contours)

            if roundest_contour is not None:
                x, y, w, h = cv2.boundingRect(roundest_contour)

                # Highlight the region of interest (ROI) on the frame
                cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi = roi_frame[y:y + h, x:x + w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                if old_gray_roi is not None and old_roi_size is not None:
                    gray_roi = cv2.resize(gray_roi, (old_roi_size[1], old_roi_size[0]))

                if p0 is None:
                    p0 = cv2.goodFeaturesToTrack(gray_roi, mask=None, maxCorners=1, qualityLevel=0.3, minDistance=7)
                    old_gray_roi = gray_roi
                    old_roi_size = gray_roi.shape
                    continue

                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_roi, gray_roi, p0, None)
                if p1 is not None and st[0][0] == 1:
                    old_point = p0[0][0]
                    new_point = p1[0][0]

                    # Draw the tracked points on the ROI
                    cv2.circle(roi_frame, (int(old_point[0]), int(old_point[1])), 5, (255, 0, 0), -1)  # Old point
                    cv2.circle(roi_frame, (int(new_point[0]), int(new_point[1])), 5, (0, 0, 255), -1)  # New point

                    vector_old = (old_point[0] - w // 2, old_point[1] - h // 2)
                    vector_new = (new_point[0] - w // 2, new_point[1] - h // 2)
                    angle = math.degrees(math.atan2(vector_new[1], vector_new[0]) -
                                         math.atan2(vector_old[1], vector_old[0]))
                    total_angle += angle

                    if abs(total_angle) >= 360.0:
                        rotation_period = frame_count / fps
                        rotation_count += 1
                        print(f"Rotation {rotation_count}: {rotation_period:.2f} seconds")
                        total_angle = 0

                    old_gray_roi = gray_roi.copy()
                    old_roi_size = gray_roi.shape
                    p0 = p1

        # Draw the central ROI boundary on the original frame
        cv2.rectangle(frame,
                      (roi_center_x - roi_width // 2, roi_center_y - roi_height // 2),
                      (roi_center_x + roi_width // 2, roi_center_y + roi_height // 2),
                      (255, 255, 0), 2)

        # Display text and visualizations
        cv2.putText(frame, f"Rotations: {rotation_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Rotating Object Visualization', frame)
        cv2.imshow('Mask Visualization', mask)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    video_path = 'Task 3/video.mov'  # Update with your video path
    measurePeriod(video_path)

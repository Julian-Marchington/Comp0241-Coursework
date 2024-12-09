import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Cloud Range Tuner')
cv2.createTrackbar('H_min', 'Cloud Range Tuner', 85, 179, nothing)
cv2.createTrackbar('H_max', 'Cloud Range Tuner', 140, 179, nothing)
cv2.createTrackbar('S_min', 'Cloud Range Tuner', 5, 255, nothing)
cv2.createTrackbar('S_max', 'Cloud Range Tuner', 80, 255, nothing)
cv2.createTrackbar('V_min', 'Cloud Range Tuner', 180, 255, nothing)
cv2.createTrackbar('V_max', 'Cloud Range Tuner', 255, 255, nothing)

while True:
    h_min = cv2.getTrackbarPos('H_min', 'Cloud Range Tuner')
    h_max = cv2.getTrackbarPos('H_max', 'Cloud Range Tuner')
    s_min = cv2.getTrackbarPos('S_min', 'Cloud Range Tuner')
    s_max = cv2.getTrackbarPos('S_max', 'Cloud Range Tuner')
    v_min = cv2.getTrackbarPos('V_min', 'Cloud Range Tuner')
    v_max = cv2.getTrackbarPos('V_max', 'Cloud Range Tuner')

    lower_clouds = np.array([h_min, s_min, v_min])
    upper_clouds = np.array([h_max, s_max, v_max])

    cloud_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
    cv2.imshow('Cloud Mask', cloud_mask)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cv2.destroyAllWindows()
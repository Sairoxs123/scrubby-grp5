import cv2
import numpy as np

current_markers = {}

# 1. Load your high-accuracy calibration data
def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix

mtx, dist = load_coefficients("../camera calibration/calibration_data.yml")

# 2. Initialize the newer ArUco API
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
# Tweak parameters for screens: lowers the threshold to find corners in bright glare
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0) # HP w200

while True:
    ret, frame = cap.read()
    if not ret: break

    # Detection
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Draw the green boxes and IDs
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Calculate Pose for EVERY detected marker (no ID filtering)
        # Use 0.05 if your phone marker is roughly 5cm wide
        obj_points = np.array([[0, 0, 0], [0.05, 0, 0], [0.05, 0.05, 0], [0, 0.05, 0]], dtype=np.float32)

        for i, corner in enumerate(corners):
            _, rvec, tvec = cv2.solvePnP(obj_points, corner, mtx, dist)
            # Draw the XYZ Axes (Red, Green, Blue)
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
            # Save marker ID and coordinates to current_markers dictionary
            current_markers[ids[i][0]] = tvec.ravel()
            if 12 in current_markers and 0 in current_markers:
                pos_origin = current_markers[12]  # Marker 12 is the origin
                pos_point = current_markers[0]     # Marker 0 is the 2nd point
                relative_pos = pos_point - pos_origin
                print(f"Relative Position of Marker 0 to Marker 12: {relative_pos}")

                distance = np.linalg.norm(relative_pos)
                print(f"Distance from Marker 12 to Marker 0: {distance*100:.2f} cm")
            #print(f"Detected ID: {ids[i][0]} at {tvec.ravel()}")

    # Draw rejected candidates in PINK (to see if glare is the issue)
    cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(255, 0, 255))

    cv2.imshow('Scrubby Lab Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
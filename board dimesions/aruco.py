import cv2
import numpy as np

current_markers = {}

# Load calibration
def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix

mtx, dist = load_coefficients("../camera calibration/calibration_data.yml")

# ArUco setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)

# Marker size (5 cm)
MARKER_SIZE = 0.10  # meters

# Object points for pose estimation
obj_points = np.array([
    [0, 0, 0],
    [MARKER_SIZE, 0, 0],
    [MARKER_SIZE, MARKER_SIZE, 0],
    [0, MARKER_SIZE, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            _, rvec, tvec = cv2.solvePnP(obj_points, corner, mtx, dist)
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

            marker_id = ids[i][0]
            current_markers[marker_id] = tvec.ravel()

        # Check if all 4 corner markers are detected
        required_ids = [0, 1, 2, 3]
        if all(mid in current_markers for mid in required_ids):

            p0 = current_markers[3]  # top right
            p1 = current_markers[2]  # bottom right
            p2 = current_markers[1]  # bottom left
            p3 = current_markers[0]  # top left

            # Width (left ↔ right)
            width_top = np.linalg.norm(p0 - p3)
            width_bottom = np.linalg.norm(p1 - p2)
            width = (width_top + width_bottom) / 2

            # Height (top ↕ bottom)
            height_right = np.linalg.norm(p0 - p1)
            height_left = np.linalg.norm(p3 - p2)
            height = (height_right + height_left) / 2

            print(f"Whiteboard Width : {width * 100:.2f} cm")
            print(f"Whiteboard Height: {height * 100:.2f} cm")
            print("-" * 40)

    cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(255, 0, 255))
    cv2.imshow("Whiteboard Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
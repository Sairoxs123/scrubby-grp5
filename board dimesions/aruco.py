import cv2
import numpy as np

# Load calibration
def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return camera_matrix, dist_matrix

mtx, dist = load_coefficients(r"C:\Users\Sai20\Desktop\Sai Teja\Scrubby\camera calibration\calibration_data.yml")

# ArUco setup
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)

# CRITICAL: Measure the BLACK SQUARE ONLY with a ruler.
# If it is 8.5cm, this MUST be 0.085.
MARKER_SIZE = 0.10  # meters

# FIX 2: Define origin (0,0,0) exactly at the center of the marker
half_size = MARKER_SIZE / 2.0
obj_points = np.array([
    [-half_size, -half_size, 0], # Top-Left
    [ half_size, -half_size, 0], # Top-Right
    [ half_size,  half_size, 0], # Bottom-Right
    [-half_size,  half_size, 0]  # Bottom-Left
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FIX 3: Clear the dictionary EVERY frame so we don't mix old/new coordinates
    current_markers = {}

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

            # These points are now the exact geometric centers of the markers
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

            # Note: This is the center-to-center distance.
            # If the markers are inside the board, the true board is slightly larger.
            print(f"Center-to-Center Width : {width * 100:.2f} cm")
            print(f"Center-to-Center Height: {height * 100:.2f} cm")
            print("-" * 40)

    cv2.imshow("Whiteboard Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
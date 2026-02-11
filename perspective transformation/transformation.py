import cv2
import numpy as np
import cv2.aruco as aruco

def load_calibration(yaml_file):
    """Loads camera matrix and distortion coefficients from an OpenCV YAML file."""
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Unable to open calibration file: {yaml_file}")

    mtx_node = fs.getNode("K")
    dist_node = fs.getNode("D")
    if mtx_node.empty() or dist_node.empty():
        fs.release()
        raise ValueError("Calibration file missing 'K' or 'D' nodes.")

    mtx = mtx_node.mat()
    dist = dist_node.mat()
    fs.release()
    return mtx, dist

def run_live_scrubby(marker_size_cm=20.0):
    # 1. Load Calibration and Setup ArUco
    try:
        mtx, dist = load_calibration("../camera calibration/calibration_data.yml")
        print("Calibration data loaded successfully.")
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    cap = cv2.VideoCapture(0) # Use 0 for default webcam
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    print("Starting live feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame using your calibration data
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_mtx)

        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) >= 4:
            marker_corners = {ids[i][0]: corners[i][0] for i in range(len(ids))}

            # Use IDs based on your sheet: 0(TL), 1(TR), 3(BR), 2(BL)
            required_ids = [0, 1, 2, 3]
            if all(mid in marker_corners for mid in required_ids):

                # Source Points (Outer Corners)
                src_pts = np.float32([
                    marker_corners[0][0], # TL
                    marker_corners[1][1], # TR
                    marker_corners[3][2], # BR
                    marker_corners[2][3]  # BL
                ])

                # Calculate Scale and Dimensions
                marker_px_w = np.linalg.norm(marker_corners[0][0] - marker_corners[0][1])
                ratio = marker_size_cm / marker_px_w

                real_w = np.linalg.norm(src_pts[0] - src_pts[1]) * ratio
                real_h = np.linalg.norm(src_pts[0] - src_pts[3]) * ratio

                # Dynamic Aspect Ratio for Warped Window
                target_w = 800
                target_h = int(real_h * (target_w / real_w))

                dst_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])

                # Transformation
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(undistorted_frame, M, (target_w, target_h))

                # Display dimensions on the original frame
                cv2.putText(undistorted_frame, f"Board: {real_w:.1f}x{real_h:.1f} cm",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Warped Whiteboard (ROI)", warped)

        cv2.imshow("Original Feed (Undistorted)", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_scrubby(marker_size_cm=20.0)
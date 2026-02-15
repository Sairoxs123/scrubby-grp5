import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# --- Configuration ---
model_path = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\train5\weights\best.pt'
model = YOLO(model_path)
MARKER_SIZE_CM = 10.0  # Physical size of your markers

def extract_whiteboard(frame):
    # Using the ArUco detector setup from your teammate's code
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) >= 4:
        ids = ids.flatten()
        marker_centers = {}
        marker_pixel_widths = []

        for i, marker_id in enumerate(ids):
            m_corners = corners[i][0]
            # Center calculation
            center = np.mean(m_corners, axis=0)
            marker_centers[marker_id] = center
            # Width for scaling: Distance between two corners of one marker
            pixel_w = np.linalg.norm(m_corners[0] - m_corners[1])
            marker_pixel_widths.append(pixel_w)

        required_ids = [0, 1, 2, 3] # TL, TR, BL, BR mapping
        if all(req_id in marker_centers for req_id in required_ids):
            # Order from teammate's code: TL(0), TR(3), BL(1), BR(2)
            src_pts = np.array([
                marker_centers[0], marker_centers[3],
                marker_centers[1], marker_centers[2]
            ], dtype="float32")

            output_width, output_height = 1000, 700
            dst_pts = np.array([[0, 0], [output_width, 0], [0, output_height], [output_width, output_height]], dtype="float32")

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_image = cv2.warpPerspective(frame, matrix, (output_width, output_height))

            # Scale Factor: 10cm / [Avg Pixels of marker]
            avg_px_w = np.mean(marker_pixel_widths)
            cm_per_pixel = MARKER_SIZE_CM / avg_px_w

            return warped_image, src_pts, cm_per_pixel

    return None, None, None

# --- Main Execution Loop ---
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    whiteboard_view, corners_used, ratio = extract_whiteboard(frame)

    if whiteboard_view is not None:
        # Run YOLO inference on the ROI
        results = model.predict(whiteboard_view, conf=0.5, stream=True)

        for r in results:
            for box in r.boxes:
                # Get box coordinates
                # xyxy = [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().astype(int)
                # xywh = [center_x, center_y, width, height]
                c = box.xywh[0].cpu().numpy()

                # Calculate real-world coordinates
                x_cm = c[0] * ratio
                y_cm = c[1] * ratio

                # DRAWING THE BOXES ON THE ROI STREAM
                cv2.rectangle(whiteboard_view, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

                # Display CM coordinates on top of the box
                label = f"{x_cm:.1f}, {y_cm:.1f} cm"
                cv2.putText(whiteboard_view, label, (b[0], b[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Target at: {label}")

        cv2.imshow("Corrected Whiteboard ROI", whiteboard_view)

    cv2.imshow("Original Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
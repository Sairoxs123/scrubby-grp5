import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- Configuration ---
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\train\weights\best.pt'
CONFIDENCE_THRESHOLD = 0.5
MARKER_SIZE_CM = 10.0  # Physical size of ArUco markers

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. EXTRACT ROI
        whiteboard_view, ratio = extract_whiteboard_logic(frame)

        if whiteboard_view is not None:
            # 2. PREPROCESS FOR AI (Removed CLAHE/Gray)
            # We use the raw BGR warped view directly
            ai_input = whiteboard_view.copy()

            # 3. INFERENCE
            results = model.predict(ai_input, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 4. DATA PROCESSING
            connectivity_payload = []

            for r in results:
                for box in r.boxes:
                    # Get coordinates and class info
                    b = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf_score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    # Keep calculations for the robot/connectivity team
                    payload = {
                        "class": class_name,
                        "x1_cm": round(b[0] * ratio, 2),
                        "y1_cm": round(b[1] * ratio, 2),
                        "x2_cm": round(b[2] * ratio, 2),
                        "y2_cm": round(b[3] * ratio, 2),
                        "conf": round(conf_score, 2)
                    }
                    connectivity_payload.append(payload)

                    # VISUAL FEEDBACK: Show Confidence and Class Name
                    cv2.rectangle(whiteboard_view, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (0, 255, 0), 2)

                    # Formatting text to show "Class: 85%"
                    label = f"{class_name}: {conf_score:.2%}"
                    cv2.putText(whiteboard_view, label, (int(b[0]), int(b[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Performance monitoring
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(whiteboard_view, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if connectivity_payload:
                print(f"BATCH TARGETS: {connectivity_payload}")

            cv2.imshow("Scrubby - Detection View", whiteboard_view)

        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

def extract_whiteboard_logic(frame):
    """ArUco-based ROI extraction."""
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
            marker_centers[marker_id] = np.mean(m_corners, axis=0)
            marker_pixel_widths.append(np.linalg.norm(m_corners[0] - m_corners[1]))

        required_ids = [0, 1, 2, 3]
        if all(req_id in marker_centers for req_id in required_ids):
            src_pts = np.array([marker_centers[0], marker_centers[3],
                                marker_centers[1], marker_centers[2]], dtype="float32")

            output_w, output_h = 1000, 700
            dst_pts = np.array([[0,0], [output_w,0], [0,output_h], [output_w,output_h]], dtype="float32")

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, matrix, (output_w, output_h))

            ratio = MARKER_SIZE_CM / np.mean(marker_pixel_widths)
            return warped, ratio

    return None, None

if __name__ == "__main__":
    main()
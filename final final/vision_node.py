import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import math

# ==========================================
# 1. CONFIGURATION & TRUE PHYSICAL CONSTANTS
# ==========================================
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\rtdetr-l_v1_optimized\weights\best.pt'
CONFIDENCE_THRESHOLD = 0.5

# Whiteboard Physics (cm) and Warp Resolution (pixels)
BOARD_WIDTH_CM = 105.0
BOARD_HEIGHT_CM = 75.0
WARP_W_PX = 1000
WARP_H_PX = 700

# Absolute Ratios (Bypasses 3D camera scaling bugs)
RATIO_X = BOARD_WIDTH_CM / float(WARP_W_PX)  # 0.105 cm/px
RATIO_Y = BOARD_HEIGHT_CM / float(WARP_H_PX) # ~0.107 cm/px

# Robot Mechanical Physics (cm)
ROBOT_MARKER_ID = 42
OFFSET_FORWARD_CM = -6.5  # Robot center is 6.5cm behind the 7cm marker's center
OFFSET_RIGHT_CM = 1.5     # Robot center is 1.5cm to the right of the 7cm marker's center

def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def merge_all_boxes(boxes):
    def is_overlapping(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])
    def merge_boxes(b1, b2):
        return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

    merged = True
    while merged:
        merged = False
        new_boxes = []
        while boxes:
            box = boxes.pop(0)
            has_merged = False
            for i in range(len(boxes)):
                if is_overlapping(box, boxes[i]):
                    boxes.append(merge_boxes(box, boxes.pop(i)))
                    has_merged = True
                    merged = True
                    break
            if not has_merged: new_boxes.append(box)
        boxes = new_boxes
    return boxes

class StabilizationFilter:
    def __init__(self, observation_window=30, min_hits=15):
        self.window = observation_window
        self.min_hits = min_hits
        self.frame_count = 0
        self.candidates = []

    def update_and_get_batch(self, current_boxes):
        self.frame_count += 1
        for c_box in current_boxes:
            matched = False
            for cand in self.candidates:
                if get_iou(c_box, cand['box']) > 0.3:
                    # Exponential Moving Average for smoothing
                    cand['box'] = [(cand['box'][i] * 0.8) + (c_box[i] * 0.2) for i in range(4)]
                    cand['hits'] += 1
                    matched = True
                    break
            if not matched: self.candidates.append({'box': list(c_box), 'hits': 1})

        if self.frame_count >= self.window:
            valid_boxes = [c['box'] for c in self.candidates if c['hits'] >= self.min_hits]
            self.frame_count, self.candidates = 0, []
            return valid_boxes
        return None

def format_patches_for_robot(stable_batch):
    """Formats clean pixel boxes into the Connectivity Team's physical JSON schema."""
    payload = []
    for box in stable_batch:
        x1, y1, x2, y2 = box
        cm_x1 = float(round(x1 * RATIO_X, 2))
        cm_x2 = float(round(x2 * RATIO_X, 2))

        # Convert image-space Y (origin at top-left, +down) to board-space Y (origin at bottom-left, +up).
        cm_y_top = float(round((WARP_H_PX - y1) * RATIO_Y, 2))
        cm_y_bottom = float(round((WARP_H_PX - y2) * RATIO_Y, 2))

        payload.append({
            "corners": [
                {"x": cm_x1, "y": cm_y_bottom}, {"x": cm_x2, "y": cm_y_bottom},
                {"x": cm_x2, "y": cm_y_top}, {"x": cm_x1, "y": cm_y_top}
            ]
        })
    return {"patches": payload}


# ==========================================
# 3. VISION & KINEMATICS LOGIC
# ==========================================
def extract_whiteboard(frame):
    """Extracts the 1000x700 warped whiteboard view using ArUco corners."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) >= 4:
        marker_centers = {ids.flatten()[i]: np.mean(corners[i][0], axis=0) for i in range(len(ids))}
        if all(req_id in marker_centers for req_id in [0, 1, 2, 3]):
            src_pts = np.array([marker_centers[0], marker_centers[3],
                                marker_centers[1], marker_centers[2]], dtype="float32")
            dst_pts = np.array([[0,0], [WARP_W_PX,0], [0,WARP_H_PX], [WARP_W_PX,WARP_H_PX]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            return cv2.warpPerspective(frame, matrix, (WARP_W_PX, WARP_H_PX))
    return None

def get_robot_pose(warped_image):
    """
    Finds the robot in the WARPED image, calculates its true center using
    the mechanical offsets, and returns its physical (X, Y) cm and angle.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(warped_image)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ROBOT_MARKER_ID:
                c = corners[i][0] # [TL, TR, BR, BL]

                # 1. Get Marker Center in CM with board-space origin at bottom-left.
                marker_cm_x = np.mean(c[:, 0]) * RATIO_X
                marker_cm_y = (WARP_H_PX - np.mean(c[:, 1])) * RATIO_Y

                # 2. Calculate Heading Angle
                front_mid_x = (c[0][0] + c[1][0]) / 2
                front_mid_y = (c[0][1] + c[1][1]) / 2
                back_mid_x = (c[3][0] + c[2][0]) / 2
                back_mid_y = (c[3][1] + c[2][1]) / 2

                dx = front_mid_x - back_mid_x
                dy = back_mid_y - front_mid_y
                theta = math.atan2(dy, dx)

                # 3. Apply 2D Rotation Matrix for Offsets
                true_center_x = marker_cm_x + (OFFSET_FORWARD_CM * math.cos(theta) - OFFSET_RIGHT_CM * math.sin(theta))
                true_center_y = marker_cm_y + (OFFSET_FORWARD_CM * math.sin(theta) + OFFSET_RIGHT_CM * math.cos(theta))

                angle_degs = np.degrees(theta) % 360

                return {
                    "x": round(true_center_x, 2),
                    "y": round(true_center_y, 2),
                    "angle": round(angle_degs, 1)
                }
    return None


# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = StabilizationFilter(observation_window=30, min_hits=15)
    prev_time = 0

    # NOTE: Your teammates will replace these print statements with their POST requests

    while True:
        ret, frame = cap.read()
        if not ret: break

        warped = extract_whiteboard(frame)

        if warped is not None:

            # --- 1. TRACK THE ROBOT ---
            robot_state = get_robot_pose(warped)
            if robot_state:
                # Teammates: POST to /telemetry here
                # print(f"Robot Telemetry: {robot_state}")

                # Draw the True Center on the screen for the Demo
                rx = int(robot_state["x"] / RATIO_X)
                ry = int(WARP_H_PX - (robot_state["y"] / RATIO_Y))
                cv2.circle(warped, (rx, ry), 5, (255, 0, 255), -1)
                cv2.putText(warped, f"{robot_state['angle']} deg", (rx+10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # --- 2. DETECT THE INK ---
            results = model.predict(warped.copy(), conf=CONFIDENCE_THRESHOLD, verbose=False)
            raw_boxes = [
                box.xyxy[0].cpu().numpy().tolist()
                for r in results
                for box in r.boxes
                if float(box.conf[0]) > CONFIDENCE_THRESHOLD
            ]
            merged_boxes = merge_all_boxes(raw_boxes)

            # --- 3. STABILIZE & DISPATCH ---
            stable_batch = tracker.update_and_get_batch(merged_boxes)

            if stable_batch is not None:
                payload = format_patches_for_robot(stable_batch)

                # Teammates: POST to /patches here
                if payload["patches"]:
                    print(f"\n--- DISPATCHING STABLE BATCH ({len(stable_batch)} PATCHES) ---")
                    print(json.dumps(payload, indent=2))
                    for idx, patch in enumerate(payload["patches"], start=1):
                        print(f"Patch {idx} corners (cm): {patch['corners']}")

            # --- VISUAL DEBUGGING ---
            for c in tracker.candidates:
                if c['hits'] > 5:
                    bx = c['box']
                    cv2.rectangle(warped, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
                    cv2.putText(
                        warped,
                        f"Hits: {c['hits']}/{tracker.window}",
                        (int(bx[0]), int(bx[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(
                warped,
                f"FPS: {int(fps)} | Frame {tracker.frame_count}/{tracker.window}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Scrubby - Vision Node", warped)

        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
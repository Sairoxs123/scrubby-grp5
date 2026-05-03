import cv2
import numpy as np
import json
import requests
import time
import math
from ultralytics import YOLO
from websocket import create_connection

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
# FILE PATHS
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\rtdetr-l_v1_optimized\weights\best.pt'
CALIBRATION_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\camera calibration\calibration_data.yml'

# AI / NETWORKING
CONFIDENCE_THRESHOLD = 0.60
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/robot_position"

# WHITEBOARD PHYSICS
BOARD_WIDTH_CM = 105.0
BOARD_HEIGHT_CM = 75.0
WARP_W_PX = 1000
WARP_H_PX = 700
RATIO_X = BOARD_WIDTH_CM / float(WARP_W_PX)
RATIO_Y = BOARD_HEIGHT_CM / float(WARP_H_PX)

# ROBOT MECHANICAL OFFSETS
ROBOT_MARKER_ID = 4
OFFSET_FORWARD_CM = -2.0
OFFSET_RIGHT_CM = 7.0

# ROBOT MASKING FOOTPRINT (22cm x 19cm, shifted 4cm forward to cover arm)
MASK_LENGTH_CM = 22.0
MASK_WIDTH_CM = 19.0
MASK_SHIFT_FORWARD_CM = 4.0

# SPATIAL EXCLUSION ZONES (ArUco Corners)
DZ_SZ = 50
DEAD_ZONES = [
    [0, 0, DZ_SZ, DZ_SZ],                                     # Top-Left
    [WARP_W_PX - DZ_SZ, 0, WARP_W_PX, DZ_SZ],                 # Top-Right
    [0, WARP_H_PX - DZ_SZ, DZ_SZ, WARP_H_PX],                 # Bottom-Left
    [WARP_W_PX - DZ_SZ, WARP_H_PX - DZ_SZ, WARP_W_PX, WARP_H_PX] # Bottom-Right
]


# ==========================================
# 2. HELPER LOGIC
# ==========================================
def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dist = cv_file.getNode("D").mat()
    cv_file.release()
    return mtx, dist

def is_overlapping(box1, box2):
    return not (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1])

def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def merge_all_boxes(boxes):
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

def transform_coords(x_px, y_px):
    cm_x = max(0.0, min(BOARD_WIDTH_CM, x_px * RATIO_X))
    cm_y = max(0.0, min(BOARD_HEIGHT_CM, BOARD_HEIGHT_CM - (y_px * RATIO_Y)))
    return float(round(cm_x, 2)), float(round(cm_y, 2))

def format_patches_for_robot(stable_batch):
    payload = []
    for box in stable_batch:
        x1, y1, x2, y2 = box
        cm_x1, cm_y1 = transform_coords(x1, y1)
        cm_x2, cm_y2 = transform_coords(x2, y2)
        payload.append({
            "corners": [
                {"x": cm_x1, "y": cm_y1}, {"x": cm_x2, "y": cm_y1},
                {"x": cm_x2, "y": cm_y2}, {"x": cm_x1, "y": cm_y2}
            ]
        })
    return {"patches": payload}

def extract_whiteboard(frame):
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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(warped_image)

    if ids is None: return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id != ROBOT_MARKER_ID: continue

        c = corners[i][0]
        marker_px_x = float(np.mean(c[:, 0]))
        marker_px_y = float(np.mean(c[:, 1]))

        front_mid_px = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)
        back_mid_px  = ((c[3][0] + c[2][0]) / 2, (c[3][1] + c[2][1]) / 2)
        dx_px = front_mid_px[0] - back_mid_px[0]
        dy_px = front_mid_px[1] - back_mid_px[1]
        theta_px = math.atan2(dy_px, dx_px)

        fwd_x, fwd_y = math.cos(theta_px), math.sin(theta_px)
        rgt_x, rgt_y = -math.sin(theta_px), math.cos(theta_px)

        true_px_x = marker_px_x + (OFFSET_FORWARD_CM / RATIO_X) * fwd_x + (OFFSET_RIGHT_CM  / RATIO_X) * rgt_x
        true_px_y = marker_px_y + (OFFSET_FORWARD_CM / RATIO_Y) * fwd_y + (OFFSET_RIGHT_CM  / RATIO_Y) * rgt_y

        true_cx = true_px_x * RATIO_X
        true_cy = BOARD_HEIGHT_CM - (true_px_y * RATIO_Y)
        true_cx = max(0.0, min(BOARD_WIDTH_CM, true_cx))
        true_cy = max(0.0, min(BOARD_HEIGHT_CM, true_cy))

        world_theta_deg = math.degrees(math.atan2(-fwd_y, fwd_x))
        compass_angle = (90 - world_theta_deg) % 360

        return {
            "telemetry": {"x": round(true_cx, 2), "y": round(true_cy, 2), "angle": round(compass_angle, 1)},
            "internal_px": {"cx": true_px_x, "cy": true_px_y, "fwd": (fwd_x, fwd_y), "rgt": (rgt_x, rgt_y)}
        }
    return None

# ==========================================
# 3. MAIN LOOP
# ==========================================
def main():
    print("Loading Camera Calibration...")
    mtx, dist = load_coefficients(CALIBRATION_PATH)

    print(f"Loading AI model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = StabilizationFilter(observation_window=30, min_hits=15)
    prev_time = 0
    loop_count = 0

    # -> NEW: CACHE FOR CURRENTLY PROCESSING BOXES
    dispatched_boxes = []

    ws = None
    try:
        ws = create_connection(WS_URL)
        print("[OK] Connected to Connectivity WebSocket.")
    except Exception as e:
        print(f"[FAIL] WebSocket connection failed: {e}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        loop_count += 1

        frame = cv2.undistort(frame, mtx, dist, None, mtx)
        warped = extract_whiteboard(frame)

        if warped is not None:
            ai_input = warped.copy()
            display_frame = warped.copy()

            # --- Visual UI ---
            cv2.circle(display_frame, (20, 680), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, "(0, 0) Origin", (35, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

            for zone in DEAD_ZONES:
                cv2.rectangle(display_frame, (zone[0], zone[1]), (zone[2], zone[3]), (0, 0, 255), 1)

            # 3. ROBOT TRACKING & MASKING
            robot_data = get_robot_pose(warped)

            if robot_data:
                telemetry = robot_data["telemetry"]
                internal = robot_data["internal_px"]

                if ws:
                    try: ws.send(json.dumps(telemetry))
                    except: pass

                if loop_count % 15 == 0:
                    print(f"[TELEMETRY] X: {telemetry['x']:05.2f} | Y: {telemetry['y']:05.2f} | Ang: {telemetry['angle']:05.1f}°")

                cx = internal["cx"] + (MASK_SHIFT_FORWARD_CM / RATIO_X) * internal["fwd"][0]
                cy = internal["cy"] + (MASK_SHIFT_FORWARD_CM / RATIO_Y) * internal["fwd"][1]
                hl, hw = MASK_LENGTH_CM / 2.0, MASK_WIDTH_CM / 2.0

                corners_cm = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
                pts = []
                for df, dr in corners_cm:
                    px = cx + (df / RATIO_X) * internal["fwd"][0] + (dr / RATIO_X) * internal["rgt"][0]
                    py = cy + (df / RATIO_Y) * internal["fwd"][1] + (dr / RATIO_Y) * internal["rgt"][1]
                    pts.append([int(px), int(py)])

                pts_arr = np.array([pts], dtype=np.int32)

                cv2.fillPoly(ai_input, pts_arr, (255, 255, 255))

                cv2.drawContours(display_frame, pts_arr, 0, (0, 0, 0), 2)
                cv2.circle(display_frame, (int(internal["cx"]), int(internal["cy"])), 5, (255, 0, 255), -1)
                cv2.putText(display_frame, f"X:{telemetry['x']} Y:{telemetry['y']} Ang:{telemetry['angle']}deg",
                            (int(internal["cx"]) + 10, int(internal["cy"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # 4. INFERENCE & SPATIAL EXCLUSION
            is_ready = True
            try:
                resp = requests.get(f"{API_BASE_URL}/ready", timeout=0.1).json()
                is_ready = bool(resp) if not isinstance(resp, dict) else resp.get("ready", False)
            except:
                pass

            if is_ready:
                results = model.predict(ai_input, conf=CONFIDENCE_THRESHOLD, verbose=False)

                valid_raw_boxes = []
                for r in results:
                    for box in r.boxes:
                        if float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                            b = box.xyxy[0].cpu().numpy().tolist()
                            hits_dead_zone = any(is_overlapping(b, zone) for zone in DEAD_ZONES)
                            if not hits_dead_zone:
                                valid_raw_boxes.append(b)

                merged_boxes = merge_all_boxes(valid_raw_boxes)
                stable_batch = tracker.update_and_get_batch(merged_boxes)

                if stable_batch is not None and len(stable_batch) > 0:
                    payload = format_patches_for_robot(stable_batch)
                    print(f"\n--- DISPATCHING STABLE BATCH ({len(stable_batch)} PATCHES) ---")
                    try:
                        requests.post(f"{API_BASE_URL}/patches", json=payload, timeout=0.5)
                        print(f"Dispatched! Robot should start moving.")

                        # -> NEW: Save the dispatched boxes to our cache
                        dispatched_boxes = stable_batch.copy()
                    except Exception as e:
                        print(f"Failed to dispatch: {e}")

            # 5. UI VISUALIZATION (Dynamic Logic)
            if not is_ready and dispatched_boxes:
                # If the robot is busy, display what it is currently processing in Yellow
                for bx in dispatched_boxes:
                    cv2.rectangle(display_frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 255), 2)
                    cv2.putText(display_frame, "PROCESSING", (int(bx[0]), int(bx[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # If the robot is ready (idle), display the active tracking candidates in Red
                for c in tracker.candidates:
                    if c['hits'] > 5:
                        bx = c['box']
                        cv2.rectangle(display_frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Hits: {c['hits']}/{tracker.window}", (int(bx[0]), int(bx[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(display_frame, f"FPS: {int(fps)} | Tracker: {tracker.frame_count}/{tracker.window} | Ready: {is_ready}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Scrubby - Integrated Vision Node", display_frame)

        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ws: ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

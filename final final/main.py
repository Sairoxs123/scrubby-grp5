import cv2
import numpy as np
import json
import requests
import time
import math
from ultralytics import YOLO
from websocket import create_connection # pip install websocket-client

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\rtdetr-l_v1_optimized\weights\best.pt'
CONFIDENCE_THRESHOLD = 0.65 # Tuned to ignore dusters/smudges

# Networking Endpoints
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/robot_position"

# Whiteboard Physics (cm) and Warp Resolution (pixels)
BOARD_WIDTH_CM = 105.0
BOARD_HEIGHT_CM = 75.0
WARP_W_PX = 1000
WARP_H_PX = 700

RATIO_X = BOARD_WIDTH_CM / float(WARP_W_PX)  # 0.105 cm/px
RATIO_Y = BOARD_HEIGHT_CM / float(WARP_H_PX) # ~0.107 cm/px

# --- ROBOT MECHANICAL OFFSETS ---
# Driven like a wide Zamboni (20cm leading edge)
ROBOT_MARKER_ID = 4
OFFSET_FORWARD_CM = -2.0  # Center is 2cm back from the marker's center
OFFSET_RIGHT_CM = 7.0     # Center is 7cm right of the marker's center

# --- ROBOT MASKING FOOTPRINT ---
# The physical chassis is 10cm long x 20cm wide.
# We define a larger mask (e.g., 18cm long) to cover the arm, and 24cm wide for a safety buffer.
MASK_LENGTH_CM = 19.0
MASK_WIDTH_CM = 22.0
# We shift the mask's center forward from the chassis center to cover the front arm
MASK_SHIFT_FORWARD_CM = 4.0


# ==========================================
# 2. HELPER LOGIC
# ==========================================
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
    """Converts pixels to CM, inverts Y for Bottom-Left origin, and clamps bounds."""
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

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id != ROBOT_MARKER_ID:
            continue

        c = corners[i][0]

        # --- All geometry in pixel space (Y increases downward) ---
        marker_px_x = float(np.mean(c[:, 0]))
        marker_px_y = float(np.mean(c[:, 1]))

        # ArUco corner order: [TL, TR, BR, BL]
        front_mid_px = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)
        back_mid_px  = ((c[3][0] + c[2][0]) / 2, (c[3][1] + c[2][1]) / 2)

        dx_px = front_mid_px[0] - back_mid_px[0]
        dy_px = front_mid_px[1] - back_mid_px[1]

        theta_px = math.atan2(dy_px, dx_px)

        # Unit vectors for the ROBOT in pixel space
        fwd_x = math.cos(theta_px)
        fwd_y = math.sin(theta_px)
        rgt_x = -math.sin(theta_px)
        rgt_y =  math.cos(theta_px)

        # Apply mechanical offsets entirely in pixel space
        true_px_x = marker_px_x + (OFFSET_FORWARD_CM / RATIO_X) * fwd_x + (OFFSET_RIGHT_CM  / RATIO_X) * rgt_x
        true_px_y = marker_px_y + (OFFSET_FORWARD_CM / RATIO_Y) * fwd_y + (OFFSET_RIGHT_CM  / RATIO_Y) * rgt_y

        # Convert to CM and invert Y ONCE for bottom-left origin
        true_cx = true_px_x * RATIO_X
        true_cy = BOARD_HEIGHT_CM - (true_px_y * RATIO_Y)

        # Clamp to board bounds
        true_cx = max(0.0, min(BOARD_WIDTH_CM, true_cx))
        true_cy = max(0.0, min(BOARD_HEIGHT_CM, true_cy))

        # Compass Angle for Connectivity Team
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
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = StabilizationFilter(observation_window=30, min_hits=15)
    prev_time = 0
    loop_count = 0

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

        warped = extract_whiteboard(frame)

        if warped is not None:
            # Create two separate buffers: one for the AI, one for the Screen
            ai_input = warped.copy()
            display_frame = warped.copy()

            # ---------------------------------------------------------
            # A. PREPROCESSING & MASKING
            # ---------------------------------------------------------
            # 1. Mask Static ArUco Corners (White Boxes drawn ONLY on AI Input)
            mask_sz = 120
            cv2.rectangle(ai_input, (0, 0), (mask_sz, mask_sz), (255, 255, 255), -1)
            cv2.rectangle(ai_input, (WARP_W_PX - mask_sz, 0), (WARP_W_PX, mask_sz), (255, 255, 255), -1)
            cv2.rectangle(ai_input, (0, WARP_H_PX - mask_sz), (mask_sz, WARP_H_PX), (255, 255, 255), -1)
            cv2.rectangle(ai_input, (WARP_W_PX - mask_sz, WARP_H_PX - mask_sz), (WARP_W_PX, WARP_H_PX), (255, 255, 255), -1)

            # Draw visual Origin on the Display Frame ONLY
            cv2.circle(display_frame, (20, 680), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, "(0, 0) Origin", (35, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

            # 2. Extract Robot Pose
            robot_data = get_robot_pose(warped)

            if robot_data:
                telemetry = robot_data["telemetry"]
                internal = robot_data["internal_px"]

                # Send Telemetry
                if ws:
                    try: ws.send(json.dumps(telemetry))
                    except: pass

                if loop_count % 15 == 0:
                    print(f"[TELEMETRY] X: {telemetry['x']:05.2f} | Y: {telemetry['y']:05.2f} | Ang: {telemetry['angle']:05.1f}°")

                # --- VECTOR BOUNDING BOX MASKING ---
                cx = internal["cx"] + (MASK_SHIFT_FORWARD_CM / RATIO_X) * internal["fwd"][0]
                cy = internal["cy"] + (MASK_SHIFT_FORWARD_CM / RATIO_Y) * internal["fwd"][1]
                hl = MASK_LENGTH_CM / 2.0
                hw = MASK_WIDTH_CM / 2.0

                corners_cm = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
                pts = []
                for df, dr in corners_cm:
                    px = cx + (df / RATIO_X) * internal["fwd"][0] + (dr / RATIO_X) * internal["rgt"][0]
                    py = cy + (df / RATIO_Y) * internal["fwd"][1] + (dr / RATIO_Y) * internal["rgt"][1]
                    pts.append([int(px), int(py)])

                pts_arr = np.array([pts], dtype=np.int32)

                # BLIND THE AI: Draw solid white polygon ONLY on the ai_input buffer
                cv2.fillPoly(ai_input, pts_arr, (255, 255, 255))

                # SHOW THE USER: Draw black outline & UI telemetry ONLY on the display_frame buffer
                cv2.drawContours(display_frame, pts_arr, 0, (0, 0, 0), 2)
                cv2.circle(display_frame, (int(internal["cx"]), int(internal["cy"])), 5, (255, 0, 255), -1)
                coord_text = f"X:{telemetry['x']} Y:{telemetry['y']} Ang:{telemetry['angle']}deg"
                cv2.putText(display_frame, coord_text, (int(internal["cx"]) + 10, int(internal["cy"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ---------------------------------------------------------
            # B. INFERENCE & DISPATCH (Using ai_input)
            # ---------------------------------------------------------
            is_ready = False
            try:
                resp = requests.get(f"{API_BASE_URL}/ready", timeout=0.1).json()
                if isinstance(resp, dict):
                    is_ready = resp.get("ready", False)
                else:
                    is_ready = bool(resp)
            except:
                pass

            if is_ready:
                # The model runs on the pristine ai_input (Robot and ArUcos masked in pure white, NO UI TEXT!)
                results = model.predict(ai_input, conf=CONFIDENCE_THRESHOLD, verbose=False)
                raw_boxes = [box.xyxy[0].cpu().numpy().tolist() for r in results for box in r.boxes if float(box.conf[0]) > CONFIDENCE_THRESHOLD]
                merged_boxes = merge_all_boxes(raw_boxes)
                stable_batch = tracker.update_and_get_batch(merged_boxes)

                if stable_batch is not None and len(stable_batch) > 0:
                    payload = format_patches_for_robot(stable_batch)
                    print(f"\n--- DISPATCHING STABLE BATCH ({len(stable_batch)} PATCHES) ---")
                    print(json.dumps(payload, indent=2))
                    try:
                        requests.post(f"{API_BASE_URL}/patches", json=payload, timeout=0.5)
                        print(f"Dispatched! Robot should start moving.")
                    except Exception as e:
                        print(f"Failed to dispatch: {e}")

            # ---------------------------------------------------------
            # C. VISUALIZATION (Drawn on display_frame)
            # ---------------------------------------------------------
            for c in tracker.candidates:
                if c['hits'] > 5:
                    bx = c['box']
                    cv2.rectangle(display_frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Hits: {c['hits']}/{tracker.window}", (int(bx[0]), int(bx[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(display_frame, f"FPS: {int(fps)} | Tracker: {tracker.frame_count}/{tracker.window} | Ready: {is_ready}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # We display the dirty frame to you, while the AI stays blissfully blind!
            cv2.imshow("Scrubby - Integrated Vision Node", display_frame)

        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ws: ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
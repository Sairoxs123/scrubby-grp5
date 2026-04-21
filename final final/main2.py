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

# Robot Mechanical Offsets (cm)
# Driven like a wide Zamboni (20cm leading edge)
ROBOT_MARKER_ID = 4
OFFSET_FORWARD_CM = -7.0  # Center is 7cm behind the marker's front edge
OFFSET_RIGHT_CM = 0.0     # Centered horizontally

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
        front_mid_px = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)  # top edge = front
        back_mid_px  = ((c[3][0] + c[2][0]) / 2, (c[3][1] + c[2][1]) / 2)  # bottom edge = back

        dx_px = front_mid_px[0] - back_mid_px[0]
        dy_px = front_mid_px[1] - back_mid_px[1]   # ← pure pixel space, NO inversion here

        theta_px = math.atan2(dy_px, dx_px)   # angle of "forward" in pixel space

        # Unit vectors in pixel space
        # forward: (cos θ, sin θ)
        # right (90° CW in screen coords where Y-down): (-sin θ, cos θ)
        fwd_x = math.cos(theta_px)
        fwd_y = math.sin(theta_px)
        rgt_x = -math.sin(theta_px)
        rgt_y =  math.cos(theta_px)

        # Apply mechanical offsets entirely in pixel space
        true_px_x = marker_px_x + (OFFSET_FORWARD_CM / RATIO_X) * fwd_x \
                                  + (OFFSET_RIGHT_CM  / RATIO_X) * rgt_x
        true_px_y = marker_px_y + (OFFSET_FORWARD_CM / RATIO_Y) * fwd_y \
                                  + (OFFSET_RIGHT_CM  / RATIO_Y) * rgt_y

        # Convert to CM and invert Y ONCE for bottom-left origin
        true_cx = true_px_x * RATIO_X
        true_cy = BOARD_HEIGHT_CM - (true_px_y * RATIO_Y)

        # Clamp to board bounds
        true_cx = max(0.0, min(BOARD_WIDTH_CM, true_cx))
        true_cy = max(0.0, min(BOARD_HEIGHT_CM, true_cy))

        # World-space angle: flip Y component for bottom-left origin, and subtract 90 degrees
        # to match the robot's mechanical orientation to the marker.
        world_theta = math.atan2(-dy_px, dx_px) - (math.pi / 2)

        return {
            "x": round(true_cx, 2),
            "y": round(true_cy, 2),
            "angle": round(math.degrees(world_theta) % 360, 1)
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
    loop_count = 0 # Used to throttle terminal logging

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
            # ---------------------------------------------------------
            # A. PREPROCESSING & MASKING (Draw masks directly on warped)
            # ---------------------------------------------------------
            # 1. Mask ArUco Corners (White Boxes)
            mask_sz = 120

            # Visual Indicator for Bottom Left Origin (Drawn on top of the mask)
            cv2.circle(warped, (20, 680), 10, (0, 255, 0), -1)
            cv2.putText(warped, "(0, 0) Origin", (35, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

            # 2. Get Pose & Mask the Robot
            robot_state = get_robot_pose(frame) # We pass the unmasked frame to find the marker!
            if robot_state:
                # Send Telemetry
                if ws:
                    try: ws.send(json.dumps(robot_state))
                    except: pass

                # Terminal Logging (Twice a second at ~30fps)
                if loop_count % 15 == 0:
                    print(f"[TELEMETRY] X: {robot_state['x']:05.2f} | Y: {robot_state['y']:05.2f} | Ang: {robot_state['angle']:05.1f}°")

                # Revert Bottom-Left Y inversion just for drawing on the screen
                draw_y_cm = BOARD_HEIGHT_CM - robot_state["y"]
                rx_px, ry_px = int(robot_state["x"] / RATIO_X), int(draw_y_cm / RATIO_Y)

                # Draw pure white circle to mask the robot + eraser arm (13cm radius)
                # radius_px = int(13.0 / RATIO_X)
                # cv2.circle(warped, (rx_px, ry_px), radius_px, (255, 255, 255), -1)

                # Draw Visual Debugging ON TOP of the white mask
                cv2.circle(warped, (rx_px, ry_px), 5, (255, 0, 255), -1) # Magenta center dot
                coord_text = f"X:{robot_state['x']} Y:{robot_state['y']} Ang:{robot_state['angle']}deg"
                cv2.putText(warped, coord_text, (rx_px + 10, ry_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ---------------------------------------------------------
            # B. INFERENCE & DISPATCH
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
                # AI now looks at the warped image, which ALREADY has the white boxes/circles drawn on it!
                results = model.predict(warped, conf=CONFIDENCE_THRESHOLD, verbose=False)
                raw_boxes = [box.xyxy[0].cpu().numpy().tolist() for r in results for box in r.boxes if float(box.conf[0]) > CONFIDENCE_THRESHOLD]
                merged_boxes = merge_all_boxes(raw_boxes)
                stable_batch = tracker.update_and_get_batch(merged_boxes)

                if stable_batch is not None and len(stable_batch) > 0:
                    payload = format_patches_for_robot(stable_batch)
                    print(f"\n--- DISPATCHING STABLE BATCH ({len(stable_batch)} PATCHES) ---")
                    try:
                        requests.post(f"{API_BASE_URL}/patches", json=payload, timeout=0.5)
                        print(f"Dispatched! Robot should start moving.")
                    except Exception as e:
                        print(f"Failed to dispatch: {e}")

            # ---------------------------------------------------------
            # C. VISUALIZATION
            # ---------------------------------------------------------
            for c in tracker.candidates:
                if c['hits'] > 5:
                    bx = c['box']
                    cv2.rectangle(warped, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
                    cv2.putText(warped, f"Hits: {c['hits']}/{tracker.window}", (int(bx[0]), int(bx[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(warped, f"FPS: {int(fps)} | Tracker: {tracker.frame_count}/{tracker.window} | Ready: {is_ready}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Scrubby - Integrated Vision Node", warped)

        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if ws: ws.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
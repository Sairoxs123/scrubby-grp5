import cv2
import numpy as np
import json
from ultralytics import YOLO
import time

# --- Helper Functions for Merging Boxes ---
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (
        x1_max < x2_min or
        x2_max < x1_min or
        y1_max < y2_min or
        y2_max < y1_min
    )

def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return [
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max)
    ]

def merge_all_boxes(boxes):
    merged = True
    while merged:
        merged = False
        new_boxes = []

        while boxes:
            box = boxes.pop(0)
            has_merged = False

            for i in range(len(boxes)):
                if is_overlapping(box, boxes[i]):
                    new_box = merge_boxes(box, boxes[i])
                    boxes.pop(i)
                    boxes.append(new_box)
                    has_merged = True
                    merged = True
                    break

            if not has_merged:
                new_boxes.append(box)

        boxes = new_boxes
    return boxes

# --- NEW: Stabilization Filter ---
def get_iou(boxA, boxB):
    """Calculates Intersection over Union to see if two boxes are the same patch."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class StabilizationFilter:
    def __init__(self, observation_window=30, min_hits=15):
        # Watch for 30 frames (~1-2 seconds depending on FPS)
        self.observation_window = observation_window
        # Target must be seen in at least 15 of those frames to be "valid"
        self.min_hits = min_hits
        self.frame_count = 0
        self.candidates = []

    def update_and_get_batch(self, current_boxes):
        self.frame_count += 1

        # 1. Match current frame's boxes to our ongoing candidates
        for c_box in current_boxes:
            matched = False
            for candidate in self.candidates:
                if get_iou(c_box, candidate['box']) > 0.3: # 30% overlap means it's the same ink
                    # Moving average to smooth out coordinate jitter
                    candidate['box'][0] = (candidate['box'][0] * 0.8) + (c_box[0] * 0.2)
                    candidate['box'][1] = (candidate['box'][1] * 0.8) + (c_box[1] * 0.2)
                    candidate['box'][2] = (candidate['box'][2] * 0.8) + (c_box[2] * 0.2)
                    candidate['box'][3] = (candidate['box'][3] * 0.8) + (c_box[3] * 0.2)
                    candidate['hits'] += 1
                    matched = True
                    break

            # If it's a completely new box, add it to candidates
            if not matched:
                self.candidates.append({'box': list(c_box), 'hits': 1})

        # 2. Check if the observation window is over
        if self.frame_count >= self.observation_window:
            # Filter out the "ghosts/flickers" that didn't meet the min_hits threshold
            valid_boxes = [c['box'] for c in self.candidates if c['hits'] >= self.min_hits]

            # Reset tracker for the next batch
            self.frame_count = 0
            self.candidates = []

            return valid_boxes

        # Return None if we are still observing
        return None

# --- Configuration ---
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\rtdetr-l_v1_optimized\weights\best.pt'
CONFIDENCE_THRESHOLD = 0.5
MARKER_SIZE_CM = 10.0

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0
    # Initialize our filter (Watches 30 frames, requires 15 hits)
    tracker = StabilizationFilter(observation_window=30, min_hits=15)

    while True:
        ret, frame = cap.read()
        if not ret: break

        whiteboard_view, ratio = extract_whiteboard_logic(frame)

        if whiteboard_view is not None:
            ai_input = whiteboard_view.copy()
            results = model.predict(ai_input, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # 1. Collect YOLO boxes
            raw_boxes = []
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy()
                    if float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                        raw_boxes.append([b[0], b[1], b[2], b[3]])

                        # Optional: Draw raw flickering YOLO boxes in faint green for debugging
                        # cv2.rectangle(whiteboard_view, (int(b[0]), int(b[1])),
                        #               (int(b[2]), int(b[3])), (0, 255, 0), 1)

            # 2. Merge overlapping raw boxes
            merged_boxes = merge_all_boxes(raw_boxes)

            # 3. Pass to the Stabilization Filter
            stable_batch = tracker.update_and_get_batch(merged_boxes)

            # 4. If the tracker has finished a 30-frame window, dispatch the batch
            if stable_batch is not None:
                connectivity_payload = []
                for box in stable_batch:
                    x1, y1, x2, y2 = box

                    patch = {
                        "corners": [
                            {"x": float(round(x1 * ratio, 2)), "y": float(round(y1 * ratio, 2))},
                            {"x": float(round(x2 * ratio, 2)), "y": float(round(y1 * ratio, 2))},
                            {"x": float(round(x2 * ratio, 2)), "y": float(round(y2 * ratio, 2))},
                            {"x": float(round(x1 * ratio, 2)), "y": float(round(y2 * ratio, 2))}
                        ]
                    }
                    connectivity_payload.append(patch)

                output = {"patches": connectivity_payload}

                # Only print/send if there are actually stable targets
                if output["patches"]:
                    print(f"\n--- DISPATCHING STABLE BATCH ---\n{json.dumps(output, indent=2)}")

            # Visual Feedback: Draw the active tracking candidates in RED
            for c in tracker.candidates:
                if c['hits'] > 5: # Only draw if it's starting to look stable
                    bx = c['box']
                    cv2.rectangle(whiteboard_view, (int(bx[0]), int(bx[1])),
                                  (int(bx[2]), int(bx[3])), (0, 0, 255), 2)
                    cv2.putText(whiteboard_view, f"Hits: {c['hits']}/30", (int(bx[0]), int(bx[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Performance monitoring
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(whiteboard_view, f"FPS: {int(fps)} | Frame {tracker.frame_count}/30", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
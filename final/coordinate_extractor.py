import cv2
import json
from ultralytics import YOLO

# -----------------------------
# Helper: Check overlap
# -----------------------------
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (
        x1_max < x2_min or
        x2_max < x1_min or
        y1_max < y2_min or
        y2_max < y1_min
    )

# -----------------------------
# Helper: Merge two boxes
# -----------------------------
def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return [
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max)
    ]

# -----------------------------
# Merge all overlapping boxes
# -----------------------------
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

# -----------------------------
# 1. Load YOLO Model
# -----------------------------
model = YOLO("best.pt")  # your trained model

# -----------------------------
# 2. Load Image
# -----------------------------
image = cv2.imread("whiteboard.jpg")
if image is None:
    raise Exception("Image not found")

image = cv2.resize(image, (800, 600))

# -----------------------------
# 3. Run YOLO Detection
# -----------------------------
results = model(image)

boxes = []

for r in results:
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])

        # Confidence threshold
        if conf > 0.5:
            boxes.append([x1, y1, x2, y2])

# -----------------------------
# 4. Merge overlapping boxes
# -----------------------------
merged_boxes = merge_all_boxes(boxes)

# -----------------------------
# 5. Convert to JSON format
# -----------------------------
patches = []

for box in merged_boxes:
    x1, y1, x2, y2 = box

    corners = [
        {"x": int(x1), "y": int(y1)},
        {"x": int(x2), "y": int(y1)},
        {"x": int(x2), "y": int(y2)},
        {"x": int(x1), "y": int(y2)}
    ]

    patches.append({"corners": corners})

output = {"patches": patches}

# -----------------------------
# 6. Print Output
# -----------------------------
print(json.dumps(output, indent=4))
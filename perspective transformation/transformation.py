import cv2
import numpy as np
import os
from datetime import datetime

def extract_whiteboard(frame):
    # 1. Load the ArUco dictionary (The image you uploaded uses 4x4 markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 2. Detect the markers in the frame
    corners, ids, rejected = detector.detectMarkers(frame)

    # We need all 4 markers (IDs 0, 1, 2, 3) to create the whiteboard
    if ids is not None and len(ids) >= 4:
        # Flatten the IDs list
        ids = ids.flatten()

        # Create a dictionary to map ID -> Marker Center Point
        marker_centers = {}

        for i, marker_id in enumerate(ids):
            # 'corners' is a list of lists. corners[i][0] gives the 4 points of the marker.
            # We calculate the mean (average) to find the center of the marker.
            marker_corners = corners[i][0]
            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])
            marker_centers[marker_id] = (center_x, center_y)

        # Check if we found all specific IDs needed for the corners
        required_ids = [0, 1, 2, 3] # TL, TR, BL, BR
        if all(req_id in marker_centers for req_id in required_ids):

            # 3. Define the Source Points (Where the markers are in the camera frame)
            # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
            src_pts = np.array([
                marker_centers[0], # Top-Left
                marker_centers[3], # Top-Right
                marker_centers[1], # Bottom-Left
                marker_centers[2]  # Bottom-Right
            ], dtype="float32")

            # 4. Define Destination Points (The flat 'whiteboard' view)
            # We pick a fixed resolution for the output, e.g., 1000x700 pixels
            output_width = 1000
            output_height = 700

            dst_pts = np.array([
                [0, 0],                         # Top-Left
                [output_width, 0],              # Top-Right
                [0, output_height],             # Bottom-Left
                [output_width, output_height]   # Bottom-Right
            ], dtype="float32")

            # 5. Perform the Perspective Transform
            # This matrix maps the src_pts to the dst_pts
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Warp the image
            warped_image = cv2.warpPerspective(frame, matrix, (output_width, output_height))

            return warped_image, src_pts

    # Return None if we can't find the board
    return None, None

# --- Main Execution Loop ---
output_dir = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(output_dir, exist_ok=True)
image_index = 0
cap = cv2.VideoCapture(0) # Open default webcam
last_whiteboard_view = None

while True:
    ret, frame = cap.read()
    if not ret: break

    # Try to extract the whiteboard
    whiteboard_view, corners_used = extract_whiteboard(frame)
    if whiteboard_view is not None:
        last_whiteboard_view = whiteboard_view

    # Visualization: Draw the detected area on the original frame for debugging
    if corners_used is not None:
        # Convert to integer for drawing
        pts = corners_used.astype(int)
        # Draw lines connecting the markers (0->1, 1->3, 3->2, 2->0 to form a box)
        # Note: Our order in src_pts was TL(0), TR(1), BL(2), BR(3)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2) # Top
        cv2.line(frame, tuple(pts[1]), tuple(pts[3]), (0, 255, 0), 2) # Right
        cv2.line(frame, tuple(pts[3]), tuple(pts[2]), (0, 255, 0), 2) # Bottom
        cv2.line(frame, tuple(pts[2]), tuple(pts[0]), (0, 255, 0), 2) # Left

    cv2.imshow("Original Camera", frame)

    if last_whiteboard_view is not None:
        cv2.imshow("Corrected Whiteboard ROI", last_whiteboard_view)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        print(f"Key pressed: {key}")

    # Press SPACE or 's' to save the ROI image
    if key in (32, ord('s')):
        if last_whiteboard_view is None:
            print("ROI not found; nothing to save.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"roi_{timestamp}_{image_index:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, last_whiteboard_view)
            image_index += 1
            print(f"Saved: {filepath}")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO
import time

# --- Configuration ---
# Update this path to your best trained model
MODEL_PATH = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\train5\weights\best.pt'
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to detect an object

def main():
    # Load the trained YOLO model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Initialize video capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Set camera resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting video stream... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run YOLO inference on the frame
        # stream=True is efficient for video sources
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Visualize the results on the frame
        # The 'plot()' method draws bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('Ink Patch Detection - Realtime', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

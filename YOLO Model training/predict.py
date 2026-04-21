from ultralytics import YOLO
import os
import glob

# Load the trained model
# Note: Adjust the path 'runs/detect/train5/weights/best.pt' if your latest run is different
model_path = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\yolo11n_v3\weights\best.pt'
model = YOLO(model_path)

# Define path to test images
test_images_path = r'Scrubby-3\test\images'

# Run inference on the source
results = model.predict(source=test_images_path, save=True, conf=0.5)

print(f"Predictions completed. Check the 'runs/detect/predict' folder for the output images.")

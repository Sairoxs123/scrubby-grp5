from roboflow import Roboflow
from ultralytics import YOLO
import torch
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == '__main__':
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA is NOT available. Using CPU.")
        device = 'cpu'

    # Download dataset
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("scrubby-ekqbm").project("scrubby")
    version = project.version(2)
    dataset = version.download("yolo26")

    # Initialize YOLO model (using a pretrained model like yolov8n.pt or yolov11n.pt if available)
    # Since the download format was 'yolov11', we can use YOLOv11 small or nano model
    model = YOLO('yolo26n.pt')  # Starts from a pretrained model

    # Train the model
    # point to the data.yaml file in the downloaded dataset location
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=200,
        imgsz=640,
        device=device,
        plots=True,
        workers=2,
        project="Scrubby",
        name="yolo26n_v2"
    )

'''
Validation Results for v11 m:

mAP@50: 0.8178

mAP@50-95: 0.5064

Precision: 0.7761

Recall: 0.8417
'''

'''
Validation Results for v11 n:
mAP@50: 0.8445
mAP@50-95: 0.5240
Precision: 0.8695
Recall: 0.8333
'''
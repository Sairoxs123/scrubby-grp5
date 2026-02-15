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
    version = project.version(1)
    dataset = version.download("yolov11")

    # Initialize YOLO model (using a pretrained model like yolov8n.pt or yolov11n.pt if available)
    # Since the download format was 'yolov11', we can use YOLOv11 small or nano model
    model = YOLO('yolo11n.pt')  # Starts from a pretrained model

    # Train the model
    # point to the data.yaml file in the downloaded dataset location
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        device=device,
        plots=True,
        workers=2
    )

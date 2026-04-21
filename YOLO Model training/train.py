from roboflow import Roboflow
from ultralytics import YOLO
import torch
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA is NOT available. Using CPU.")
        device = 'cpu'

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("scrubby-ekqbm").project("scrubby")
    version = project.version(3)
    dataset = version.download("yolov11")
    model = YOLO('rtdetr-l.pt')

    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=2,            # Lowered from default 16 to fit VRAM
        device=device,
        workers=2,
        project="Scrubby",
        name="rtdetr-l_v1_optimized",
        cache=True,      # Turn off caching if system RAM is also tight
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
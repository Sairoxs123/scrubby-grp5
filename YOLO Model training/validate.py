from ultralytics import YOLO

if __name__ == "__main__":
    # Load the trained model
    model_path = r"C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\Scrubby\yolo11n_v2\weights\best.pt"
    model = YOLO(model_path)

    # Validate the model
    # This runs evaluation on the data specified in the 'val' key of the data.yaml file
    # We previously set 'val' to point to the test images in data.yaml
    metrics = model.val(data='Scrubby-3/data.yaml', split='val', imgsz=640)

    # Print formatted results
    print("\nValidation Results:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

'''
RT DETR Results:

Validation Results:
mAP@50: 0.8354
mAP@50-95: 0.5512
Precision: 0.8477
Recall: 0.8589
'''

'''
YOLO v11 v3 Results:

Validation Results:
mAP@50: 0.8229
mAP@50-95: 0.5458
Precision: 0.8688
Recall: 0.8121
'''


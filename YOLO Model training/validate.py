from ultralytics import YOLO

if __name__ == "__main__":
    # Load the trained model
    model_path = r'C:\Users\Sai20\Desktop\Sai Teja\Scrubby\runs\detect\train5\weights\best.pt'
    model = YOLO(model_path)

    # Validate the model
    # This runs evaluation on the data specified in the 'val' key of the data.yaml file
    # We previously set 'val' to point to the test images in data.yaml
    metrics = model.val(data='Scrubby-1/data.yaml', split='val')

    # Print formatted results
    print("\nValidation Results:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

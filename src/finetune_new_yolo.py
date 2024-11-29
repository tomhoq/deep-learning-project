from ultralytics import YOLO
import os

##
#vim ~/.config/Ultralytics/settings.json
##

print("[+] Started ...")

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

yaml_path = os.path.join(os.getenv('BLACKHOLE'), 'yolo', 'yolo.yaml')
results = model.train(data=yaml_path, epochs=100, imgsz=768)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
path = os.path.join(os.getenv('BLACKHOLE'), "from_train_set.jpg")
results = model(path)

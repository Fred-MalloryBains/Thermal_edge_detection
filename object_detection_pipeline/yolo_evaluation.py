from ultralytics import YOLO
import os
from pathlib import Path

model = YOLO("yolov8m.pt")


data_path = Path("/Volumes/Pluggable_1TB/thermal_images/archive/").expanduser()

thermal_paths = list(data_path.glob("set*/V*/lwir"))
visible_paths = list(data_path.glob("set*/V*/visible"))

        
def get_image_files(path_name):
    files = []
    for dir in path_name:
        if dir.is_dir():
            files = list(dir.iterdir()) 
            files.extend(files)
    return files

        
def extract_detections(results, model):
    r = results[0]
    
    if r.boxes is None:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    
    detections = []
    confidence = []
    
    for i in range(len(boxes)):
        detections.append({
            "class_id": int(classes[i]),
            "class_name": model.names[int(classes[i])],
            "confidence": float(conf[i]),
            "bbox": boxes[i]
        })
        confidence.append(float(conf[i]))
        
    average_confidence = sum([d["confidence"] for d in detections]) / len(detections) if detections else 0
    print ("average confidence: ", average_confidence)
    return detections

visible_files = get_image_files(visible_paths)
thermal_files = get_image_files(thermal_paths)

results = model(visible_files[0])
detections = extract_detections(results, model)

print(detections)

results = model(thermal_files[0])
detections = extract_detections(results, model)
print(detections)
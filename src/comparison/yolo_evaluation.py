import cv2

from ultralytics import YOLO
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolov8m.pt")




        


        
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


def all_file_comparison(image_one, image_two):
    

    results = model(image_one)
    detections = extract_detections(results, model)

    print(detections)

    results = model(image_two)
    detections = extract_detections(results, model)
    print(detections)


def yolo_compare(image_one, image_two, stem, suffix):
    model = YOLO("yolov8m.pt")
    results = model(image_one)
    detections_one = extract_detections(results, model)
    annotated = results[0].plot()
    Image.fromarray(annotated).save(f"outputs/yolo_comparison/annotated_{stem}_gt.png")
    
    results = model (image_two)
    detections_two = extract_detections(results, model)
    annotated = results[0].plot()
    Image.fromarray(annotated).save(f"outputs/yolo_comparison/annotated_{stem}_{suffix}.png")
    
    
    return detections_one, detections_two

if __name__ == "__main__":
    recon_stems = ["I01035", "I02509", "I00000"]
    base = "outputs/final_comparison"
    for stem in recon_stems:
        gt    = cv2.imread(f"{base}/gt/{stem}.png")
        thermal = cv2.imread(f"{base}/thermal/{stem}.png")
        edge_base = cv2.imread(f"{base}/base_edges/{stem}_base.png")
        edge_hed = cv2.imread(f"{base}/base_edges/{stem}_model.png")
        token = cv2.imread(f"{base}/recon/{stem}_tokens.png")
        recon  = cv2.imread(f"{base}/recon/{stem}.png")
        
        print(f"Comparing GT and Recon for stem {stem}")
        yolo_compare(gt, recon, stem, 'recon')
        
        print (f"Comparing GT and Token for stem {stem}")
        yolo_compare(gt, token, stem, 'token')
        
        print (f"Comparing GT and Thermal for stem {stem}")
        yolo_compare(gt, thermal, stem, 'thermal')
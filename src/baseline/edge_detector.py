from pathlib import Path 
import cv2
import os

from scipy import ndimage
import numpy as np 
from PIL import Image

data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser()

pairs = []

for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
    
    thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
    
    if thermal_path.exists():
        pairs.append((visible_path, thermal_path))

print("Total pairs:", len(pairs))
print(pairs[:3])

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    edges = clahe.apply(blurred)
    return edges

def normalize_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def calculate_canny_percentile(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    low = np.percentile(magnitude, 80)
    high = np.percentile(magnitude, 95)
    
    edges = cv2.Canny(image, low, high)
    
    return edges
for i in range(0,10):
    print (f"Processing pair {i+1}/{len(pairs)}: {pairs[i][0].name} and {pairs[i][1].name}")
    
"""    
for i in range (0,10):
    visible_path, thermal_path = pairs[i]
    therm_img = normalize_image(thermal_path)
    vis_img = normalize_image(visible_path)
    vis_img = preprocess_image(vis_img)
    therm_img = preprocess_image(therm_img)
    thermal_edges = calculate_canny_percentile(therm_img)
    visible_edges = calculate_canny_percentile(vis_img)
    Image.fromarray(thermal_edges).save(f"outputs/edges_thermal_{i}.png")
    Image.fromarray(visible_edges).save(f"outputs/edges_visible_{i}.png")
"""
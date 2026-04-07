from pathlib import Path 
import cv2
import os

from scipy import ndimage
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt



## resolution of the thermal images?
## prompt vague and specific - middle ground
def get_pairs(data_path):
    pairs = []

    for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
        
        thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
        
        if thermal_path.exists():
            pairs.append((visible_path, thermal_path))
    return pairs[:32]

def preprocess_image_one(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    edges = clahe.apply(blurred)


    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    Image.fromarray(edges).save("outputs/1.png")
    
def preprocess_image_two(thermal_img):
    """
    Encodes 1-channel thermal into a 3-channel pseudo-RGB image 
    aligned for HED semantic detection.
    """
    # 1. Edge-Preserving Denoising
    # Bilateral filtering is superior for HED as it removes sensor grain
    # while locking the sharp object boundaries[cite: 18, 68].

    denoised = cv2.bilateralFilter(thermal_img, 9, 75, 75)
    
    # 2. Multi-Scale Contrast Extraction
    # Channel 1: Low contrast (mimics flat-light visible scenes)
    clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
    
    # Channel 2: Medium/Structural details
    clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
    
    # Channel 3: High contrast (mimics strong edges/shadows in RGB)
    # This helps the model interpret contents even with ambiguous shapes[cite: 174].
    clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
    
    # 3. Semantic Channel Stacking
    # HED expects a 3-channel input to engage its deep encoding layers[cite: 81, 88].
    pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
    
    return pseudo_rgb
    
def normalize_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized
 





def process_edge(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    (H, W) = img.shape[:2]

    img = preprocess_image_two(img)  # returns [low, mid, high] as uint8

    # HED pretrained BSDS expects ImageNet BGR means, not local image means
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=0.7,
        size=(500, 500),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False
    )

    # For plotting, clip to valid uint8 range after shifting means back
    blob_for_plot = np.moveaxis(blob[0], 0, 2)
    blob_for_plot = np.clip(blob_for_plot + 104, 0, 255).astype(np.uint8)
    plt.imshow(blob_for_plot)

    net.setInput(blob)
    hed = net.forward()
    hed = hed[0, 0, :, :]
    hed = (255 * hed).astype("uint8")
    
    margin = 40
    hed = hed[margin:H-margin, margin:W-margin]
    hed = cv2.resize(hed, (W, H))
    return hed

def run():
    pairs = get_pairs(data_path)
    for i, (visible_path, thermal_path) in enumerate(pairs[:10]):
        print (f"Processing pair {i+1}/{len(pairs)}: {visible_path.name} and {thermal_path.name}")
        edge_map = process_edge(thermal_path)
        Image.fromarray(edge_map).save(f"outputs/edges_hed/edges_hed_{thermal_path.stem}.png")



data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
protoPath = "src/preprocess/hed_model/deploy.prototxt"
modelPath = "src/preprocess/hed_model/hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

if __name__ == "__main__":
    run()
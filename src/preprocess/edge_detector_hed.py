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

for i in range(0,10):
    print (f"Processing pair {i+1}/{len(pairs)}: {pairs[i][0].name} and {pairs[i][1].name}")
 

protoPath = "src/preprocess/hed_model/deploy.prototxt"
modelPath = "src/preprocess/hed_model/hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load the input image and grab its dimensions, for future use while defining the blob
img = cv2.imread(str(pairs[0][1]), cv2.IMREAD_GRAYSCALE)  #Using the first thermal image as an example
(H, W) = img.shape[:2]

img = preprocess_image_two(img)


mean_pixel_values= np.average(img, axis = (0,1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.5, size=(W, H),
                             mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             swapRB= True, crop=True)

# set the blob as the input to the network and perform a forward pass
# to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0,0,:,:]  #Drop the other axes 
#hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")  #rescale to 0-255
Image.fromarray(hed).save("outputs/edges_hed/edges_hed_thermal_three.png")
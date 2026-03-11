from pathlib import Path 
import numpy as np 
import cv2
import os

from scipy import ndimage
import numpy as np 
from PIL import Image


for i in range (0,10):
    img_path = os.path.expanduser(f"~/Downloads/data/I0000{i}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    edges = clahe.apply(img)
    edges = cv2.Canny(edges, 50, 110)
    Image.fromarray(edges).save(f"edges_{i}.png")
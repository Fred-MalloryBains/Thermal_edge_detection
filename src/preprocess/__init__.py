from pathlib import Path 
import numpy as np 
import cv2

from scipy import ndimage
import cv2 
import numpy as np 

data_path = Path("/Volumes/Pluggable_1TB/thermal_images/archive/").expanduser()
thermal_paths = list(data_path.glob("set*/V*/lwir"))
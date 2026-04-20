import cv2
import numpy as np
from pathlib import Path

data_path = Path("/Volumes/Pluggable_1TB/thermal_images/archive/").expanduser()
thermal_paths = list(data_path.glob("set*/V*/lwir"))

all_files = []
for lwir_dir in thermal_paths:
    if lwir_dir.is_dir():
        files = list(lwir_dir.iterdir())
        all_files.extend(files)

# Pick first 9 images
img_paths = all_files[:9]

# Load in grayscale
images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in img_paths]

# Resize for consistent grid
images = [cv2.resize(img, (320, 240)) for img in images]

def nothing(x):
    pass

# Main window
cv2.namedWindow("Canny Grid")

# Trackbars
cv2.createTrackbar("Threshold1", "Canny Grid", 50, 500, nothing)
cv2.createTrackbar("Threshold2", "Canny Grid", 150, 500, nothing)


while True:
    t1 = cv2.getTrackbarPos("Threshold1", "Canny Grid")
    t2 = cv2.getTrackbarPos("Threshold2", "Canny Grid")

    # Apply Canny to each image
    edges = [cv2.Canny(img, t1, t2) for img in images]

    # Build 3×3 grid
    row1 = np.hstack(edges[0:3])
    row2 = np.hstack(edges[3:6])
    row3 = np.hstack(edges[6:9])
    grid = np.vstack((row1, row2, row3))

    cv2.imshow("Canny Grid", grid)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

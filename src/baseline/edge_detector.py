from pathlib import Path 
import cv2
import os

from scipy import ndimage
import numpy as np 
from PIL import Image


def canny_edge_detection(image):
    return cv2.Canny(image, 70, 150)

def sobel_edge_detection(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compare_edges(image_path, output_dir="outputs/comparison"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GRAYSCALE (critical fix)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    colour_image = cv2.imread(str(image_path))  # For side-by-side comparison
    # Apply methods
    sobel = sobel_edge_detection(image)
    canny = canny_edge_detection(image)

    # Save outputs
    cv2.imwrite(str(output_dir / "sobel.jpg"), sobel)
    cv2.imwrite(str(output_dir / "canny.jpg"), canny)

    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    comparison = np.hstack([colour_image, sobel_bgr, canny_bgr])


    cv2.imwrite(str(output_dir / "comparison.jpg"), comparison)

def process_image(img):
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
        clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
        pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
        
        cv2.imwrite("outputs/baseline/normal/processed_low.jpg", clahe_low)
        cv2.imwrite("outputs/baseline/normal/processed_mid.jpg", clahe_mid)
        cv2.imwrite("outputs/baseline/normal/processed_high.jpg", clahe_high)
        cv2.imwrite("outputs/baseline/normal/processed_merge.jpg", pseudo_rgb)
    
if __name__ == "__main__":
    image_path = "outputs/baseline/lwir/example_one.jpg"
    #compare_edges(image_path)
    process_image(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
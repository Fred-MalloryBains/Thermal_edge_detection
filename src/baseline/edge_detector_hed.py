from pathlib import Path 
import cv2
import os
import torch
from torchvision import transforms

from scipy import ndimage
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

from src.preprocess.run import Network  # sniklauss file for pytorch HED


def raw_transform(image):
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])(image)

def process_image(self, img):
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
        clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
        pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
        return pseudo_rgb


def process_edge(img_path, model, device):
    img = Image.open(img_path).convert("RGB")
    
    input_tensor = raw_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        edge_map = outputs[-1]
        
        edge_map = torch.sigmoid(edge_map)
        
        edge_map = edge_map.squeeze().cpu().numpy()
    
        edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

if __name__ == "__main__":
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    visible_path = "outputs/baseline/visible/I00000.jpg"
    thermal_path = "outputs/baseline/lwir/I00000.jpg"
    
    model = Network().to(device)
    model.eval()
    
    visible_edge = process_edge(visible_path, model, device)
    thermal_edge = process_edge(thermal_path, model, device)
    
    Image.fromarray(visible_edge).save("outputs/baseline/edges/edges_visible_hed.png")
    Image.fromarray(thermal_edge).save("outputs/baseline/edges/edges_thermal_hed.png")
    
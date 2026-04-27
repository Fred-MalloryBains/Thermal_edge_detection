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


def get_pairs(data_path):
    pairs = []

    for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
        
        thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
        
        if thermal_path.exists():
            pairs.append((visible_path, thermal_path))
    return pairs[:32]

def raw_transform(image):
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])(image)

def process_image(img):
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
        clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
        pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
        return pseudo_rgb


def process_edge_visible(img_path, model, device):
    img = Image.open(img_path).convert("RGB")
    
    input_tensor = raw_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        edge_map = outputs[-1]
        
        edge_map = torch.sigmoid(edge_map)
        
        edge_map = edge_map.squeeze().cpu().numpy()
    
        edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

def process_edge_thermal(img_path, model, device):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    processed_img = process_image(img)
    
    input_tensor = raw_transform(Image.fromarray(processed_img)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        edge_map = outputs[-1]
        
        edge_map = torch.sigmoid(edge_map)
        
        edge_map = edge_map.squeeze().cpu().numpy()
    
        edge_map = (edge_map * 255).astype(np.uint8)
    
    return edge_map

if __name__ == "__main__":
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    #visible_path = "outputs/baseline/visible/I00000.jpg"
    #thermal_path = "outputs/baseline/lwir/I00000.jpg"
    
    model = Network().to(device)
    model.eval()
    
    #visible_edge = process_edge(visible_path, model, device)
    #thermal_edge = process_edge(thermal_path, model, device)
    
    #Image.fromarray(visible_edge).save("outputs/baseline/edges/edges_visible_hed_pipe_2.png")
    #Image.fromarray(thermal_edge).save("outputs/baseline/edges/edges_thermal_hed_pipe_2.png")
    
    data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
    pairs = get_pairs(data_path)
    
    # Create output dir if it doesn't exist
    os.makedirs("outputs/edges_hed_comp", exist_ok=True)

    for i, (visible_path, thermal_path) in enumerate(pairs[:10]):
        print(f"Processing pair {i+1}/{len(pairs)}: {visible_path.name} and {thermal_path.name}")
        
        # Call the new PyTorch function
        gt_edge_map = process_edge_visible(visible_path, model, device)
        t_egde_map = process_edge_thermal(thermal_path, model, device)
        
        Image.fromarray(gt_edge_map).save(f"outputs/edges_hed_comp/edges_hed{visible_path.stem}_gt.png")
        Image.fromarray(t_egde_map).save(f"outputs/edges_hed_comp/edges_hed{thermal_path.stem}_thermal.png")
    
    
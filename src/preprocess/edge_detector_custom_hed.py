import torch
from pathlib import Path 
import cv2
import os
import numpy as np 
from PIL import Image
import sys

# Ensure Python can find your Network class
sys.path.insert(0, '.')
from src.preprocess.run import Network

from torchvision import transforms


# 1. GLOBAL PYTORCH INITIALIZATION

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load the custom architecture and weights
pytorch_net = Network().to(device)
weights_path = "hed_thermal.pth"

# Load weights and set to eval mode 
pytorch_net.load_state_dict(torch.load(weights_path, map_location=device))
pytorch_net.eval() 



# 2. HELPER FUNCTIONS (Unchanged)

def get_pairs(data_path):
    pairs = []

    for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
        
        thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
        
        if thermal_path.exists():
            pairs.append((visible_path, thermal_path))
    return pairs[:32]


def preprocess_image_two(thermal_img):
    denoised = cv2.bilateralFilter(thermal_img, 9, 75, 75)
    clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
    clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
    clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
    pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
    return pseudo_rgb



def raw_transform(img):
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])(img)


# 3. THE NEW PYTORCH INFERENCE FUNCTION

def process_edge_pytorch(img_path):
    # 1. Read and preprocess
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    processed_img = preprocess_image_two(img)  # Apply the same preprocessing as during training
    tensor_img = raw_transform(Image.fromarray(processed_img)).unsqueeze(0).to(device)
    
    # 3. Format for PyTorch
   
    # 4. Run Inference without calculating gradients
    with torch.no_grad():
        outputs = pytorch_net(tensor_img)
        fused = outputs[-1] if isinstance(outputs, tuple) else outputs
        fused = torch.sigmoid(fused)  # Ensure output is in [0, 1] range
        fused = torch.where(fused > 0.1, fused, torch.zeros_like(fused))

    # 5. Post-process back to OpenCV format
    # Move to CPU, remove batch/channel dims, and scale back to uint8
    edge_map = fused.squeeze().cpu().numpy()
    edge_map = (edge_map * 255.0).clip(0, 255).astype(np.uint8)
    

    return edge_map


def post_process(edge_map):
    return edge_map

# 4. EXECUTION

def run():
    data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
    pairs = get_pairs(data_path)
    
    # Create output dir if it doesn't exist
    os.makedirs("outputs/edges_hed", exist_ok=True)

    for i, (visible_path, thermal_path) in enumerate(pairs[:10]):
        print(f"Processing pair {i+1}/{len(pairs)}: {visible_path}")
        
        # Call the new PyTorch function
        #edge_map = process_edge_pytorch(thermal_path)
        #outputs = post_process(edge_map)
        #Image.fromarray(outputs).save(f"outputs/edges_hed_custom/edges_hed{thermal_path.stem}.png")

if __name__ == "__main__":
    run()
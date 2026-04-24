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


# 1. GLOBAL PYTORCH INITIALIZATION

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load the custom architecture and weights
pytorch_net = Network().to(device)
weights_path = "hed_thermal.pth"

# Load weights and set to eval mode (CRITICAL for deterministic inference)
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

def preprocess_image_one(image):
    pass




# 3. THE NEW PYTORCH INFERENCE FUNCTION

def process_edge_pytorch(img_path):
    # 1. Read and preprocess
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    original_H, original_W = img.shape[:2]
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel 

    # Returns uint8 numpy array [H, W, 3]
    pseudo_rgb = preprocess_image_two(img)  
    

    # 2. Resize to 500x500 to match previous margin logic
    resized_img = cv2.resize(pseudo_rgb, (500, 500))

    # 3. Format for PyTorch
    # The sniklaus 'forward' function expects values in [0.0, 1.0].
    # It handles multiplying by 255 and subtracting ImageNet means internally.
    tensor_img = torch.from_numpy(resized_img).float() / 255.0
    
    # Permute from [H, W, C] to [C, H, W] and add Batch dimension [1, C, H, W]
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)

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
    
    # 6. Apply margins and resize back to the thermal camera's original resolution
    margin = 40
    edge_map = edge_map[margin:500-margin, margin:500-margin]
    edge_map = cv2.resize(edge_map, (original_W, original_H))

    return edge_map



# 4. EXECUTION

def run():
    data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
    pairs = get_pairs(data_path)
    
    # Create output dir if it doesn't exist
    os.makedirs("outputs/edges_hed", exist_ok=True)

    for i, (visible_path, thermal_path) in enumerate(pairs[:10]):
        print(f"Processing pair {i+1}/{len(pairs)}: {visible_path.name} and {thermal_path.name}")
        
        # Call the new PyTorch function
        edge_map = process_edge_pytorch(thermal_path)
        
        Image.fromarray(edge_map).save(f"outputs/edges_hed_custom/edges_hed{thermal_path.stem}.png")

if __name__ == "__main__":
    run()
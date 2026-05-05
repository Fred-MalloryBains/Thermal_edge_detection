from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 

from preprocess.base_hed import Network  # sniklauss file for pytorch HED


## loads KAIST dataset and transforms the data into normalised and raw tensors
## for both textual inversion and deep learning training


class EdgeToImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size=512):
        
        self.pairs = self.get_pairs(data_path)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network().to(self.device)
        self.model.eval()
        

        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        ## data loading logic
        visible_path, thermal_path = self.pairs[idx]

        thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)  
        thermal_img = self.process_image(thermal_img)
        thermal_img = Image.fromarray(thermal_img).convert("RGB")  # conver to 3 channel array
        
        edge_map_three = self.process_edge_soft(visible_path)
        edge_map_one = Image.fromarray(edge_map_three).convert("L")  # Convert to PIL Image in grayscale
        edge_map_three = Image.fromarray(edge_map_three).convert("RGB")  # Convert to PIL Image in RGB
        
        visible_img = cv2.imread(str(visible_path))
        visible_img = Image.fromarray(visible_img).convert("RGB")
        
        ## Return tesnored paired data for textual inversion and edge map training
        return {
            'thermal_sd': self.raw_transform(thermal_img),
            'visible_sd': self.raw_transform(visible_img),
            'edge_sd': self.raw_transform(edge_map_three),
            'thermal_raw': self.raw_transform(thermal_img),
            'edge_raw': self.raw_transform(edge_map_one),
            'name' : thermal_path.parents[2].name + "_" + thermal_path.parents[1].name + "_" + thermal_path.stem
        }
    
    def get_pairs(self, data_path):
        pairs = []

        ## logic for grabbing the image pairs for KAIST, have to be modified for other datasets
        for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
            set_name = visible_path.parents[2].name
            if set_name in ["set00", "set01", "set02", "set03", "set04", "set05"]:
                thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
            
                if thermal_path.exists():
                    pairs.append((visible_path, thermal_path))
        return pairs
    
    def process_image(self, img):
        ## preprocess the thermal image as done in evaluation
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(img)
        clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(img)
        pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
        return pseudo_rgb
    
    def crop_and_resize(self, img):
        ## unused helper function to resize images
        H, W = img.shape[:2]
        margin = 40
        img = img[margin:H-margin, margin:W-margin]
        img = cv2.resize(img, (W, H))
        return img
    
    def process_edge(self, img_path):
        ## edge detection function using the custom HED model, returns raw edge map without post processing
        img = Image.open(img_path).convert("RGB")
        
        input_tensor = self.raw_transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            edge_map = outputs[-1]
            
            edge_map = torch.sigmoid(edge_map)
            
            edge_map = edge_map.squeeze().cpu().numpy()
        
            edge_map = (edge_map * 255).astype(np.uint8)
        
        return edge_map
    
    def process_edge_soft (self, img_path):
        ## preprocess the edge map with some soft post processing 
        ## to give the model more information during training, this is not used for evaluation
        edge = self.process_edge(img_path)
        
        edge = edge.astype(np.float32) / 255.0
        
        edge = cv2.GaussianBlur(edge, (3,3), 0)
        
        edge = (edge * 255).astype(np.uint8)
        return edge
    

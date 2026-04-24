from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 

from src.preprocess.run import Network  # sniklauss file for pytorch HED


## loads KAIST dataset and transforms the data into normalised and raw tensors
## for both textual inversion and deep learning training

## to-do: add prerprocessing with extra channels for thermal data (CLAHE)

#data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
#protoPath = "src/preprocess/hed_model/deploy.prototxt"
#modelPath = "src/preprocess/hed_model/hed_pretrained_bsds.caffemodel"

#net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

class EdgeToImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size=512):
        
        self.pairs = self.get_pairs(data_path)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network().to(self.device)
        self.model.eval()
        
        self.sd_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        visible_path, thermal_path = self.pairs[idx]

        thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)  
        thermal_img = self.process_image(thermal_img)
        thermal_img = Image.fromarray(thermal_img).convert("RGB")  # conver to 3 channel array
        
        edge_map_three = self.process_edge_soft(visible_path)
        edge_map_one = Image.fromarray(edge_map_three).convert("L")  # Convert to PIL Image in grayscale
        edge_map_three = Image.fromarray(edge_map_three).convert("RGB")  # Convert to PIL Image in RGB
        
        visible_img = cv2.imread(str(visible_path))
        visible_img = Image.fromarray(visible_img).convert("RGB")
        
        

        #print(f"Edge map shape: {edge_map_one.size}")
        #print(f"Thermal image shape: {thermal_img.size}")
        
        return {
            'thermal_sd': self.sd_transform(thermal_img),
            'visible_sd': self.sd_transform(visible_img),
            'edge_sd': self.sd_transform(edge_map_three),
            'thermal_raw': self.raw_transform(thermal_img),
            'edge_raw': self.raw_transform(edge_map_one),
            'name' : thermal_path.parents[2].name + "_" + thermal_path.parents[1].name + "_" + thermal_path.stem
        }
    
    def get_pairs(self, data_path):
        pairs = []

        for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
            set_name = visible_path.parents[2].name
            if set_name in ["set00", "set01", "set02"]:
                thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
            
                if thermal_path.exists():
                    pairs.append((visible_path, thermal_path))
        return pairs[:32]  # Limit to first 32 pairs for now
    
    def process_image(self, img):
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        clahe_low = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(denoised)
        clahe_mid = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        clahe_high = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(denoised)
        pseudo_rgb = cv2.merge([clahe_low, clahe_mid, clahe_high])
        return pseudo_rgb
    
    def crop_and_resize(self, img):
        H, W = img.shape[:2]
        margin = 40
        img = img[margin:H-margin, margin:W-margin]
        img = cv2.resize(img, (W, H))
        return img
    
    def process_edge(self, img_path):
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
        edge = self.process_edge(img_path)
        
        edge = edge.astype(np.float32) / 255.0
        
        kernel = np.ones((3,3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        
        edge = cv2.GaussianBlur(edge, (3,3), 0)
        
        edge = (edge * 255).astype(np.uint8)
        return edge
    

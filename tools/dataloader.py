from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 


## loads KAIST dataset and transforms the data into normalised and raw tensors
## for both textual inversion and deep learning training

## to-do: add prerprocessing with extra channels for thermal data (CLAHE)

#data_path = Path("/Volumes/Samsung_1TB/thermal_images/archive/").expanduser() 
protoPath = "src/preprocess/hed_model/deploy.prototxt"
modelPath = "src/preprocess/hed_model/hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

class EdgeToImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size=512):
        
        self.pairs = self.get_pairs(data_path)
        
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

        img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)  
        img = self.process_image(img)
        img = self.crop_and_resize(img)
        img = Image.fromarray(img, mode="RGB") # Convert to PIL Image in grayscale

        edge_map_three = self.process_edge_soft(visible_path)

        edge_map_one = Image.fromarray(edge_map_three).convert("L")  # Convert to PIL Image in grayscale

        return {
            #'thermal_sd': self.sd_transform(img),
            #'edge_sd': self.sd_transform(edge_map_three),
            'thermal_raw': self.raw_transform(img),
            'edge_raw': self.raw_transform(edge_map_one)
        }
    
    def get_pairs(self, data_path):
        pairs = []

        for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
            
            thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
            
            if thermal_path.exists():
                pairs.append((visible_path, thermal_path))
        return pairs
    
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
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        (H, W) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=0.7,
            size=(W, H), 
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False,
            crop=False
        )

        net.setInput(blob)
        hed = net.forward()
        
        # --- FIX START ---
        # hed is shape (1, 1, H, W). We need (H, W)
        hed = np.squeeze(hed) 
        # --- FIX END ---

        hed = (255 * hed).astype("uint8")

        # Now hed.shape is (H, W), so crop_and_resize will work correctly
        hed = self.crop_and_resize(hed)

        return hed
    
    def process_edge_soft (self, img_path):
        edge = self.process_edge(img_path)
        
        edge = edge.astype(np.float32) / 255.0
        
        kernel = np.ones((10,10), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        
        edge = cv2.GaussianBlur(edge, (9,9), 0)
        
        edge = (edge * 255).astype(np.uint8)
        return edge
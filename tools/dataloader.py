from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from src.preprocess.edge_detector_hed import process_edge
import numpy as np

## loads KAIST dataset and transforms the data into normalised and raw tensors
## for both textual inversion and deep learning training

## to-do: add prerprocessing with extra channels for thermal data (CLAHE)

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

        img = Image.open(thermal_path).convert("L")
        img = np.array(img)

        img = np.stack([img, img, img], axis=-1)  # fake RGB but controlled
        img = Image.fromarray(img)

        edge_map = process_edge(visible_path)

        edge_map = Image.fromarray(edge_map).convert("RGB")

        


        return {
            'thermal_sd': self.sd_transform(img),
            'edge_sd': self.sd_transform(edge_map),
            'thermal_raw': self.raw_transform(img),
            'edge_raw': self.raw_transform(edge_map)
        }
    
    def get_pairs(self, data_path):
        pairs = []

        for visible_path in data_path.glob("set*/V*/visible/*.jpg"):
            
            thermal_path = visible_path.parents[1] / "lwir" / visible_path.name
            
            if thermal_path.exists():
                pairs.append((visible_path, thermal_path))
        return pairs
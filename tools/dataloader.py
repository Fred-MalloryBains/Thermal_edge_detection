from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from src.preprocess.edge_detector_hed import get_pairs, process_edge

class EdgeToImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_size=512):
        self.pairs = get_pairs(data_path)

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.edge_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        visible_path, thermal_path = self.pairs[idx]

        img = Image.open(visible_path).convert("RGB")

        edge_map = process_edge(thermal_path)

        edge_map = Image.fromarray(edge_map).convert("RGB")

        img = self.img_transform(img)
        edge_map = self.edge_transform(edge_map)


        return edge_map, img
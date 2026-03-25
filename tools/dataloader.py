class EdgeToImageDataset(torch.utils.data.Dataset):
    def __init__(self, edge_dir, image_dir, image_size=512):
        self.edge_paths = sorted(list(Path(edge_dir).glob("*")))
        self.image_paths = sorted(list(Path(image_dir).glob("*")))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.edge_paths)

    def __getitem__(self, idx):
        edge = Image.open(self.edge_paths[idx]).convert("RGB")
        img  = Image.open(self.image_paths[idx]).convert("RGB")

        edge = self.transform(edge)
        img  = self.transform(img)

        return edge, img
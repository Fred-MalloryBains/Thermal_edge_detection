import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '.')
from src.preprocess.run import Network  # your existing file
from tools.dataloader import EdgeToImageDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = Network().to(device)
model.train()

for name, param in model.named_parameters():
    is_backbone = any(name.startswith(b) for b in [
        'netVggOne', 'netVggTwo', 'netVggThr', 'netVggFou', 'netVggFiv'
    ])
    param.requires_grad_(not is_backbone)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")  # 6K


dataset = EdgeToImageDataset(
    data_path=Path("/Volumes/Samsung_1TB/thermal_images/archive"),
    image_size=256
)

n_val = int(0.1 * len(dataset))
train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20, eta_min=1e-5)

def hed_loss(logits, gt):
    """
    Weighted Binary Cross Entropy using Logits for stability.
    Positives are weighted by (1 - beta), Negatives by beta.
    """
    mask = gt > 0.5
    pos_num = torch.sum(mask)
    neg_num = torch.sum(~mask)
    total_num = pos_num + neg_num

    # beta is the fraction of positive pixels (edges)
    # In HED, we weight negative pixels by beta and positive pixels by (1 - beta)
    beta = pos_num / total_num
    
    # Weight for edge pixels (positives) should be high
    # Weight for background pixels (negatives) should be low
    pos_weight = (1.0 - beta)
    neg_weight = beta

    # Stable BCE with Logits: 
    # loss = - [w_p * y * log(sigmoid(x)) + w_n * (1-y) * log(1-sigmoid(x))]
    loss = -(pos_weight * gt * F.logsigmoid(logits) + 
             neg_weight * (1 - gt) * F.logsigmoid(-logits))
    
    return loss.mean()

def run_epoch(loader, train=True):
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            thermal = batch['thermal_raw'].to(device, dtype=torch.float32)
            gt_edge = batch['edge_raw'].to(device, dtype=torch.float32)

            gt_edge = gt_edge[:,0:1,:,:]  # ensure gt_edge has shape [B,1,H,W]

            outputs = model(thermal)  # [B,1,H,W] sigmoid output

            loss = 0 
            for pred in outputs:
                loss += hed_loss(pred, gt_edge)

            if train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()

            total += loss.item()
    return total / len(loader)

best_val = float("inf")
for epoch in range(32):
    train_loss = run_epoch(train_dl, train=True)
    val_loss   = run_epoch(val_dl,   train=False)
    scheduler.step()
    print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "hed_thermal.pth")
        print(f"  saved (val={val_loss:.4f})")
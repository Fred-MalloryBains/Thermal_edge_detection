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
    # Unfreeze the Score layers, the Combine layer, AND the first VGG block
    is_trainable = any(name.startswith(b) for b in [
        'netScore', 'netCombine', 'netVggOne' 
    ])
    param.requires_grad_(is_trainable)

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

optimiser = torch.optim.Adam(trainable, lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20, eta_min=1e-5)

def save_debug_images(thermal, gt_edge, outputs, epoch):
    
    for i in range(min(2, thermal.size(0))):  # Save up to 2 samples per epoch
        # Save Ground Truth
        gt_np = (gt_edge[i, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(gt_np).save(f"debug/ep{epoch}_samp{i}_gt.png")
        
        # Save Side Outputs and Fused
        for j, pred in enumerate(outputs):
            # Apply sigmoid because we are training with logits
            p_map = torch.sigmoid(pred[i, 0]).cpu().detach().numpy()
            p_img = (p_map * 255).astype(np.uint8)
            
            label = f"side{j+1}" if j < 5 else "fused"
            Image.fromarray(p_img).save(f"debug/ep{epoch}_samp{i}_{label}.png")

def hed_loss(logits, gt):
    gt_hard = (gt > 0.5).float()  # Binarize ground truth for loss calculation
    mask = gt_hard.bool()
    pos_num = mask.float().sum()
    neg_num = (~mask).float().sum()
    total   = pos_num + neg_num

    # Natural class-balanced weights — don't modify these
    alpha = neg_num / total   # weight for positives (edges)
    beta  = 3 *  (pos_num / total)   # weight for negatives (background)

    loss = -(alpha * gt_hard * F.logsigmoid(logits) +
             beta  * (1 - gt_hard) * F.logsigmoid(-logits))
    return loss.mean()

def run_epoch(loader, train=True):
    model.train(train)
    total = 0.0
    has_saved = False
    with torch.set_grad_enabled(train):
        for batch in loader:
            thermal = batch['thermal_raw'].to(device, dtype=torch.float32)
            gt_edge = batch['edge_raw'].to(device, dtype=torch.float32)

            gt_edge = gt_edge[:,0:1,:,:]  # ensure gt_edge has shape [B,1,H,W]

            outputs = model(thermal)  # [B,1,H,W] sigmoid output

            loss = 0 
            weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
            for i, pred in enumerate(outputs):
                loss += weights[i] * hed_loss(pred, gt_edge)
            
            if not has_saved and not train:
                save_debug_images(thermal, gt_edge, outputs, epoch)
                has_saved = True

            if train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()

            total += loss.item()
    return total / len(loader)

best_val = float("inf")
for epoch in range(20):
    train_loss = run_epoch(train_dl, train=True)
    val_loss   = run_epoch(val_dl,   train=False)
    scheduler.step()
    print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "hed_thermal.pth")
        print(f"  saved (val={val_loss:.4f})")
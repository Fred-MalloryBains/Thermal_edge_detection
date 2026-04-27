import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '.')
from src.preprocess.run import Network  # sniklauss file for pytorch HED
from tools.dataloader import EdgeToImageDataset
from scipy.ndimage import distance_transform_edt


def init():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = Network().to(device)
    model.train()

    for name, param in model.named_parameters():
        # Unfreeze the Score layers, the Combine layer, AND the first VGG block
        is_trainable = any(name.startswith(b) for b in [
            'netScore', 'netCombine'
        ])
        param.requires_grad_(is_trainable)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")  # 6K


    optimiser = torch.optim.Adam(trainable, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=40, T_mult=1, eta_min=1e-5)
    
    return device, model, scheduler, optimiser, trainable

def save_debug_images(thermal, gt_edge, outputs, epoch, names):
    
    for i in range(min(2, thermal.size(0))):  # Save up to 2 samples per epoch
        # Save Ground Truth
        gt_np = (gt_edge[i, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(gt_np).save(f"debug/ep{epoch}_{names[i]}_gt.png")
        
        # Save Side Outputs and Fused
        for j, pred in enumerate(outputs): # only fused outptus for now
            # Apply sigmoid because we are training with logits
            p_map = torch.sigmoid(pred[i, 0]).cpu().detach().numpy()
            p_img = (p_map * 255).astype(np.uint8)
            
            label = f"side{j+1}" if j < 5 else "fused"
            if label == "fused":
                Image.fromarray(p_img).save(f"debug/ep{epoch}_{names[i]}_{label}.png")

def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, 
               alpha: float = 0.95, gamma: float = 4.0):
    
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt, reduction='none')
    
    p = torch.sigmoid(pred_logits)
    # p_t: probability of the TRUE class at each pixel
    p_t = p * gt + (1 - p) * (1 - gt)
    # alpha_t: per-pixel class weight
    alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
    
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def soft_dice_loss(pred_logits: torch.Tensor, gt: torch.Tensor,
                   smooth: float = 1.0):
    
    p = torch.sigmoid(pred_logits)
    
    # Flatten spatial dims, keep batch
    p_flat = p.view(p.size(0), -1)
    gt_flat = gt.view(gt.size(0), -1)
    
    intersection = (p_flat * gt_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (p_flat.sum(dim=1) + gt_flat.sum(dim=1) + smooth)
    
    return 1.0 - dice.mean()


def boundary_iou_loss(pred_logits, gt,
                      dilation_px: int = 4, smooth: float = 1.0):
    
    p = torch.sigmoid(pred_logits)
    
    # Build dilation kernel
    k = 2 * dilation_px + 1
    kernel = torch.ones(1, 1, k, k, device=pred_logits.device, dtype=pred_logits.dtype)
    pad = dilation_px
    
    # Dilate GT: any pixel within dilation_px of a GT edge becomes "valid"
    gt_dilated = F.conv2d(gt, kernel, padding=pad).clamp(0, 1)
    # Dilate prediction: allows pred edges to "reach" toward GT
    p_dilated  = F.conv2d(p,  kernel, padding=pad).clamp(0, 1)
    
    p_flat  = p_dilated.view(p.size(0), -1)
    gt_flat = gt_dilated.view(gt.size(0), -1)
    
    intersection = (p_flat * gt_flat).sum(dim=1)
    union        = (p_flat + gt_flat - p_flat * gt_flat).sum(dim=1)
    iou          = (intersection + smooth) / (union + smooth)
    
    return 1.0 - iou.mean()


def hed_loss(pred_logits: torch.Tensor, gt: torch.Tensor,
             device: torch.device,
             w_focal: float = 0.5,
             w_dice: float = 0.3,
             w_biou: float = 0.2):
    
    fl  = focal_loss(pred_logits, gt)
    dl  = soft_dice_loss(pred_logits, gt)
    bil = boundary_iou_loss(pred_logits, gt)
    sparsity = torch.sigmoid(pred_logits).mean()
    
    
    return w_focal * fl + w_dice * dl + w_biou * bil + 0.13 * sparsity
    
    
def run_epoch(loader, device, optimiser, model, epoch, trainable, train=True):
    model.train(train)
    total = 0.0
    has_saved = False
    with torch.set_grad_enabled(train):
        for batch in loader:
            thermal = batch['thermal_raw'].to(device, dtype=torch.float32)
            gt_edge = batch['edge_raw'].to(device, dtype=torch.float32)
            names = batch['name']

            gt_edge = gt_edge[:,0:1,:,:]  # ensure gt_edge has shape [B,1,H,W]

            outputs = model(thermal)  # [B,1,H,W] sigmoid output

            loss = 0 
            weights = [0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 1]
            for i, pred in enumerate(outputs):
                loss += weights[i] * hed_loss(pred, gt_edge, device)
            
            if not has_saved and not train:
                save_debug_images(thermal, gt_edge, outputs, epoch, names)
                has_saved = True

            if train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()

            total += loss.item()
    return total / len(loader)
        
def train(dataset): 
    device, model, scheduler, optimiser, trainable = init ()


    TRAIN_SAMPLES = 1500
    VAL_SAMPLES = 375


    rng = np.random.default_rng(seed=42)
    all_indices = np.arange(len(dataset))
    val_indices = rng.choice(all_indices, size=VAL_SAMPLES, replace=False)
    train_pool = np.setdiff1d(all_indices, val_indices)

    val_dl = DataLoader(
        dataset,
        batch_size=4,
        sampler=val_indices.tolist(),
        shuffle=False,
        num_workers=0
    )

    best_val = float("inf")
    for epoch in range(100):
        train_indices = rng.choice(train_pool, size=TRAIN_SAMPLES, replace=False).tolist()
        train_dl = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_indices,
            num_workers=0
        )
        
        train_loss = run_epoch(train_dl, device, optimiser, model, epoch, trainable, train=True)
        val_loss   = run_epoch(val_dl, device, optimiser, model, epoch, trainable, train=False)
        scheduler.step()
        print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "hed_thermal_v2.pth")
            print(f"  saved (val={val_loss:.4f})")
            
if __name__ == "__main__":
    dataset = EdgeToImageDataset(
        data_path=Path("/Volumes/Samsung_1TB/thermal_images/archive"),
        image_size=512
    )
    train (dataset)
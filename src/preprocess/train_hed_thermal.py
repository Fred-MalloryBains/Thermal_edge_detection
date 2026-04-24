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
            'netScore', 'netCombine', 'netVggOne' 
        ])
        param.requires_grad_(is_trainable)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")  # 6K


    optimiser = torch.optim.Adam(trainable, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100, eta_min=1e-5)
    
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

def hed_loss(logits, gt, device):
    mask = (gt > 0.3).float()  # edges vs background
    pos_num = mask.float().sum()
    neg_num = (~mask.bool()).float().sum()
    total   = pos_num + neg_num

    # Natural class-balanced weights — don't modify these
    alpha = neg_num / total   # weight for positives (edges)
    beta  = max(3 *  (pos_num / total), 0.4)  # weight for negatives (background)


    dist_weights = compute_distance_weights(gt, device)
    
    edge_loss = -alpha * gt * F.logsigmoid(logits)

    tv_loss = compute_tv_loss(logits)
    
    bg_penalty = 1.0 - dist_weights
    
    bg_loss = -beta * (1 - gt) * F.logsigmoid(-logits) * bg_penalty
    
    return (edge_loss + bg_loss).mean() + 0.1 * tv_loss

def compute_distance_weights(gt, device, sigma = 5):
    weights = torch.zeros_like(gt)
    for b in range(gt.shape[0]):
        mask_np = gt[b,0].cpu().numpy().astype(bool)
        if mask_np.any():
            dt = distance_transform_edt(~mask_np)
            w = np.exp(-dt/ sigma).astype(np.float32)
            w = np.clip(w, 0.1, 0.9) # avoid extreme convergence
        else:
            w = np.zeros(mask_np.shape, dtype=np.float32)
        weights[b,0] = torch.from_numpy(w)
        
    return weights.to(gt.device)

def compute_tv_loss(logits):
    # Total Variation Loss to encourage smoothness
    h_variation = torch.abs(logits[:,:,1:,:] - logits[:,:,:-1,:])
    w_variation = torch.abs(logits[:,:,:,1:] - logits[:,:,:,:-1])
    return (h_variation.mean() + w_variation.mean())
    
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
            weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1]
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


    TRAIN_SAMPLES = 100 
    VAL_SAMPLES = 20


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
    for epoch in range(150):
        train_indices = rng.choice(train_pool, size=TRAIN_SAMPLES, replace=False).tolist()
        train_dl = DataLoader(
            dataset,
            batch_size=16,
            sampler=train_indices,
            num_workers=0
        )
        
        train_loss = run_epoch(train_dl, device, optimiser, model, epoch, trainable, train=True)
        val_loss   = run_epoch(val_dl, device, optimiser, model, epoch, trainable, train=False)
        scheduler.step()
        print(f"[{epoch:02d}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "hed_thermal.pth")
            print(f"  saved (val={val_loss:.4f})")
            
if __name__ == "__main__":
    dataset = EdgeToImageDataset(
        data_path=Path("/Volumes/Samsung_1TB/thermal_images/archive"),
        image_size=512
    )
    train (dataset)
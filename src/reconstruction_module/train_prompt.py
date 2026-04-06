import torch 
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from tools.dataloader import EdgeToImageDataset

## This script learns the prompt that reduces average loss between the generated images and ground truth images.

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# Control Net model
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-hed", 
        torch_dtype=dtype
    )
except Exception as error:
    print (error)
    exit(1)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=dtype
)


scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(20)

pipe.scheduler = scheduler
pipe = pipe.to(device)
pipe.vae.scaling_factor = 0.18215
model_dtype = next(pipe.unet.parameters()).dtype


with torch.no_grad():
    seed_tokens = pipe.tokenizer(
        "photorealistic, realistic colours, high resolution, urban scene, daylight",
        return_tensors="pt", 
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
    ).to(device)
    seed_embeddings = pipe.text_encoder(input_ids=seed_tokens.input_ids, attention_mask=seed_tokens.attention_mask).last_hidden_state

v = seed_embeddings.detach().float().requires_grad_(True)

optimiser = torch.optim.Adam([v], lr=1e-3, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=100, eta_min=1e-5
)

#@torch.no_grad()
def encode_to_latent(im_tensor):
    im_tensor = im_tensor.to(device=device, dtype=model_dtype)
    z = pipe.vae.encode(im_tensor).latent_dist.sample()
    return (z * pipe.vae.config.scaling_factor).to(dtype=model_dtype)


def training_step(edge_map, gt_img):
    B = edge_map.shape[0]
    z_gt = encode_to_latent(gt_img)

    v_cond = v.to(device=device, dtype=model_dtype).expand(B, -1, -1)

    ## sample step id
    t_idx = torch.randint(0, len(scheduler.timesteps), (B,), device=scheduler.timesteps.device)
    timesteps = scheduler.timesteps[t_idx].to(device)

    noise = torch.randn_like(z_gt)
    z_noisy = scheduler.add_noise(z_gt, noise, timesteps)

    edge_cond = edge_map.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        down_block_res, mid_block_re = controlnet(
            z_noisy, 
            timesteps,
            encoder_hidden_states=v_cond,
            controlnet_cond= edge_cond,
            return_dict=False
        )
    
    noise_pred = pipe.unet(
        z_noisy,
        timesteps,
        encoder_hidden_states=v_cond,
        down_block_additional_residuals=down_block_res,
        mid_block_additional_residual=mid_block_re,
    ).sample.float()

    z_pred = torch.stack([
        scheduler.step(noise_pred[i], timesteps[i], z_noisy[i]).prev_sample
        for i in range(B)
    ]
    )

    L_latent = F.mse_loss(z_pred, z_gt)
    L_reg = 0.01 * F.mse_loss(v, seed_embeddings.float().detach())
    loss = L_latent + L_reg

    return loss

def train(dataloader: DataLoader, n_epochs: int = 100):
    best_loss = float("inf")
    best_v = None

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for edge_maps, rgb_targets in dataloader:
            edge_maps   = edge_maps.to(device)
            rgb_targets = rgb_targets.to(device)

            optimiser.zero_grad()
            loss = training_step(edge_maps, rgb_targets)
            loss.backward()

            # Gradient clipping prevents v from jumping out of semantic space
            torch.nn.utils.clip_grad_norm_([v], max_norm=1.0)
            optimiser.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        lr_scheduler.step()

        if avg < best_loss:
            best_loss = avg
            best_v = v.detach().clone()
            torch.save(best_v, f"best_v_epoch{epoch}.pt")

        print(f"[{epoch:03d}] avg loss={avg:.5f}  lr={optimiser.param_groups[0]['lr']:.2e}")

    return best_v

dataset = EdgeToImageDataset(
    data_path=Path("/home/bot/dev/data/archive"),
    image_size=512
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)

best_prompt_embedding = train(dataloader, n_epochs=100)
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

from tools.dataloader import EdgeToImageDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  # force stability on MPS

try:
# Load models
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-hed",
        torch_dtype=dtype
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
except Exception as error:
    print (error)
    exit(1)

scheduler = pipe.scheduler
unet = pipe.unet
text_encoder = pipe.text_encoder
vae = pipe.vae


for p in [unet, controlnet, pipe.vae, pipe.text_encoder]:
    p.requires_grad_(False)
    p.eval()
    
tokeniser = pipe.tokenizer


## seed embeddings
with torch.no_grad():
    tokens = tokeniser(
        "photorealistic urban scene",
        return_tensors="pt",
        padding="max_length",
        max_length=tokeniser.model_max_length,
        truncation=True
    ).to(device)

    seed_emb = text_encoder(**tokens).last_hidden_state

## trainable delta
    
delta = torch.zeros_like(seed_emb, requires_grad=True)

## https://github.com/rinongal/textual_inversion/blob/main/ldm/models/diffusion/ddpm.py#L1443
optimizer = torch.optim.Adam([delta], lr=1e-3)

def training_step(edge_map, x0):

    B = x0.shape[0]

    # Encode image
    z0 = vae.encode(x0).latent_dist.sample()
    z0 = z0 * vae.config.scaling_factor

    # Sample timestep 
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B,))
    t = t.to(device)

    # Noise
    noise = torch.randn_like(z0)

    # Forward diffusion 
    zt = scheduler.add_noise(z0, noise, t)

    # Build embeddings
    e_theta = seed_emb + delta
    e_theta = e_theta.repeat(B, 1, 1)

    e_null = torch.zeros_like(e_theta)

    # ControlNet
    with torch.no_grad():
        down_c, mid_c = controlnet(
            zt, t,
            encoder_hidden_states=e_theta,
            controlnet_cond=edge_map,
            return_dict=False
        )

    # UNet prediction
    eps_cond = unet(
        zt, t,
        encoder_hidden_states=e_theta,
        down_block_additional_residuals=down_c,
        mid_block_additional_residual=mid_c
    ).sample

    eps_uncond = unet(
        zt, t,
        encoder_hidden_states=e_null
    ).sample

    # CFG
    w = 5.0
    eps = eps_uncond + w * (eps_cond - eps_uncond)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    # Reconstruct x0
    alpha_t = alphas_cumprod[t].view(-1,1,1,1)
    sqrt_alpha = alpha_t.sqrt()
    sqrt_one_minus = (1 - alpha_t).sqrt()

    x0_pred = (zt - sqrt_one_minus * eps) / sqrt_alpha

    # Losses
    L_recon = F.mse_loss(x0_pred, z0)
    L_noise = F.mse_loss(eps, noise)
    L_reg = F.mse_loss(e_theta, seed_emb)

    return L_recon + 0.1 * L_noise + 0.1 * L_reg

def train(dataloader, n_epochs=10):

    best_loss = float("inf")

    for epoch in range(n_epochs):
        total = 0

        for edge_map, x0 in dataloader:

            edge_map = edge_map.to(device, dtype=dtype)
            x0 = x0.to(device, dtype=dtype)

            optimizer.zero_grad()

            loss = training_step(edge_map, x0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_([delta], 1.0)
            optimizer.step()

            total += loss.item()

        avg = total / len(dataloader)

        print(f"[{epoch}] loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(delta.detach(), "best_delta.pt")
            
            
            
dataset = EdgeToImageDataset(
    data_path=Path("/Volumes/Samsung_1TB/thermal_images/archive"),
    image_size=128
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

best_prompt_embedding = train(dataloader, n_epochs=5)
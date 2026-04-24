import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from torch.utils.data import DataLoader
from pathlib import Path

from tools.dataloader import EdgeToImageDataset


# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32


# ----------------------------
# Load models
# ----------------------------
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-hed",
        torch_dtype=dtype
    )
except Exception as error:
    print(error)
    exit(1)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)

# Replace scheduler with DDIM (faster training)
scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(6)
pipe.scheduler = scheduler

pipe = pipe.to(device)

# Ensure correct scaling
pipe.vae.scaling_factor = 0.18215
model_dtype = next(pipe.unet.parameters()).dtype


def debug_tensor(name, x):
    print(f"{name}: requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
    

# ----------------------------
# Textual inversion setup
# ----------------------------

# Add new learnable token
placeholder_token = "<thermal>"
num_added = pipe.tokenizer.add_tokens(placeholder_token)
assert num_added == 1, "Token already exists"

pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)

# Initialise embedding from a meaningful prompt
initializer_prompt = "photorealistic, realistic colours, high resolution, urban scene, daylight"

with torch.no_grad():
    tokens = pipe.tokenizer(
        initializer_prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
    ).to(device)

    seed_embeddings = pipe.text_encoder(**tokens).last_hidden_state

# Use mean embedding as initial value
init_embedding = seed_embeddings.mean(dim=1)[0]

embedding_layer = pipe.text_encoder.get_input_embeddings()

with torch.no_grad():
    embedding_layer.weight[placeholder_token_id] = init_embedding


# ----------------------------
# Freeze all models
# ----------------------------
for p in pipe.unet.parameters():
    p.requires_grad = False

for p in controlnet.parameters():
    p.requires_grad = False

for p in pipe.vae.parameters():
    p.requires_grad = False

for p in pipe.text_encoder.parameters():
    p.requires_grad = False

# Enable gradients only on embedding matrix
embedding_layer.weight.requires_grad = True


# Optimiser
optimiser = torch.optim.Adam(
    embedding_layer.parameters(),
    lr=5e-4
)


# ----------------------------
# Helper functions
# ----------------------------

def encode_to_latent(im_tensor):
    im_tensor = im_tensor.to(device=device, dtype=model_dtype)
    z = pipe.vae.encode(im_tensor).latent_dist.sample()
    return z * pipe.vae.config.scaling_factor


def get_text_embeddings(prompts):
    tokens = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    return pipe.text_encoder(**tokens).last_hidden_state


# ----------------------------
# Training step
# ----------------------------
@torch.enable_grad()
def training_step(edge_map, gt_img):
    B = edge_map.shape[0]

    z_gt = encode_to_latent(gt_img)

    prompts = ["a <thermal> urban scene"] * B
    e_cond = get_text_embeddings(prompts)

    #debug_tensor("e_cond", e_cond)

    t_idx = torch.randint(0, len(scheduler.timesteps), (B,), device="cpu")
    timesteps = scheduler.timesteps[t_idx].to(device=device)

    noise = torch.randn_like(z_gt)
    z_noisy = scheduler.add_noise(z_gt, noise, timesteps)

    with torch.inference_mode(False):
        # Clone all inputs — tensors created outside inference_mode(False)
        # are flagged as inference tensors and can't enter autograd
        z_noisy_ = z_noisy.clone()
        e_cond_  = e_cond.clone()
        edge_    = edge_map.to(dtype=model_dtype).clone()

        down_block_res, mid_block_res = controlnet(
            z_noisy_,
            timesteps.to(dtype=model_dtype),
            encoder_hidden_states=e_cond_,
            controlnet_cond=edge_,
            return_dict=False
        )

        down_block_res = [r.clone() for r in down_block_res]
        mid_block_res  = mid_block_res.clone()

        noise_pred = pipe.unet(
            z_noisy_,
            timesteps.to(dtype=model_dtype),
            encoder_hidden_states=e_cond_,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
        ).sample

    #debug_tensor("noise_pred", noise_pred)

    loss = F.mse_loss(noise_pred.float(), noise.float())

    #debug_tensor("loss", loss)

    return loss


# ----------------------------
# Training loop
# ----------------------------

def train(dataloader, n_epochs=100):
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            edge_maps = batch["edge_sd"].to(device)
            rgb_targets = batch["visible_sd"].to(device)

            optimiser.zero_grad()

            loss = training_step(edge_maps, rgb_targets)
            loss.backward()
            

            # Mask gradients so only the placeholder token is updated
            with torch.no_grad():
                grads = embedding_layer.weight.grad
                mask = torch.zeros_like(grads)
                mask[placeholder_token_id] = 1.0
                embedding_layer.weight.grad *= mask

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(embedding_layer.parameters(), 1.0)

            optimiser.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)

        if avg < best_loss:
            best_loss = avg

            learned_embedding = embedding_layer.weight[placeholder_token_id].detach().cpu()
            torch.save(learned_embedding, "best_thermal_token.pt")

        print(f"[{epoch:03d}] avg loss={avg:.5f}")

    return embedding_layer.weight[placeholder_token_id].detach().cpu()


# ----------------------------
# Dataset
# ----------------------------

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

best_embedding = train(dataloader, n_epochs=100)
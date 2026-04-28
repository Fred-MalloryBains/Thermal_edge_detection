import torch
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from torch.utils.data import DataLoader
from pathlib import Path

from tools.dataloader import EdgeToImageDataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32

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

scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler.set_timesteps(200)
pipe.scheduler = scheduler

pipe = pipe.to(device)
pipe.vae.scaling_factor = 0.18215
model_dtype = next(pipe.unet.parameters()).dtype


# ----------------------------
# Textual inversion setup — CHANGED: multiple tokens
# ----------------------------

placeholder_tokens = ["<KAIST-1>", "<KAIST-2>", "<KAIST-3>"] 
num_added = pipe.tokenizer.add_tokens(placeholder_tokens)           
assert num_added == len(placeholder_tokens), "Tokens already exist" 

pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

placeholder_token_ids = [                                           # CHANGED
    pipe.tokenizer.convert_tokens_to_ids(t) for t in placeholder_tokens
]

embedding_layer = pipe.text_encoder.get_input_embeddings()

# initialise each token from a different word to break symmetry
init_words = ["photorealistic", "daylight", "urban"]
with torch.no_grad():
    for token_id, word in zip(placeholder_token_ids, init_words):
        word_id = pipe.tokenizer.convert_tokens_to_ids(word)
        embedding_layer.weight[token_id] = embedding_layer.weight[word_id].clone()


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

embedding_layer.weight.requires_grad = True

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


SEED_PROMPT = "scene, high resolution, 8k, sharp, structured"

# ----------------------------
# Training step
# ----------------------------
@torch.enable_grad()
def training_step(edge_map, gt_img):
    B = edge_map.shape[0]

    z_gt = encode_to_latent(gt_img)

    # CHANGED: all three tokens in the prompt
    prompts = ["<KAIST-1> <KAIST-2> <KAIST-3> " + SEED_PROMPT] * B
    e_cond = get_text_embeddings(prompts)

    t_idx = torch.randint(50, len(scheduler.timesteps), (B,), device="cpu")
    timesteps = scheduler.timesteps[t_idx].to(device=device)

    noise = torch.randn_like(z_gt)
    z_noisy = scheduler.add_noise(z_gt, noise, timesteps)

    with torch.inference_mode(False):
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

    loss = F.mse_loss(noise_pred.float(), noise.float())
    return loss


# ----------------------------
# Training loop
# ----------------------------

def train(dataloader, n_epochs=50):
    best_loss = float("inf")

    # ADDED: track embedding drift from initialisation
    initial_embeddings = [
        embedding_layer.weight[tid].detach().clone()
        for tid in placeholder_token_ids
    ]

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            edge_maps = batch["edge_sd"].to(device)
            rgb_targets = batch["visible_sd"].to(device)

            optimiser.zero_grad()
            loss = training_step(edge_maps, rgb_targets)
            loss.backward()

            # CHANGED: mask gradients for all placeholder token ids
            with torch.no_grad():
                grads = embedding_layer.weight.grad
                mask = torch.zeros_like(grads)
                for token_id in placeholder_token_ids:  # CHANGED
                    mask[token_id] = 1.0
                embedding_layer.weight.grad *= mask

            torch.nn.utils.clip_grad_norm_(embedding_layer.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)

        # ADDED: drift logging per token
        with torch.no_grad():
            for i, (tid, init_emb) in enumerate(zip(placeholder_token_ids, initial_embeddings)):
                current = embedding_layer.weight[tid]
                drift = (current - init_emb).norm().item()
                cos_sim = F.cosine_similarity(
                    current.unsqueeze(0), init_emb.unsqueeze(0)
                ).item()
                print(f"  token-{i+1} drift={drift:.4f} cos_sim={cos_sim:.4f}")

        if avg < best_loss:
            best_loss = avg
            # CHANGED: save all token embeddings
            learned_embeddings = [
                embedding_layer.weight[tid].detach().cpu()
                for tid in placeholder_token_ids
            ]
            torch.save(learned_embeddings, "weights/best_KAIST_tokens_big.pt")

        print(f"[{epoch:03d}] avg loss={avg:.5f}  best={best_loss:.5f}")


# ----------------------------
# Dataset
# ----------------------------

dataset = EdgeToImageDataset(
    data_path=Path("/Volumes/Samsung_1TB/thermal_images/archive"),
    image_size=256
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

train(dataloader, n_epochs=50)
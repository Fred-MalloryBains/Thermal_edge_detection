from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import torch
from diffusers.utils import load_image

device = "mps" if torch.backends.mps.is_available() else "cpu"
# Match training dtype exactly
dtype = torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-hed",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# ---- Reconstruct seed_emb exactly as in training ----
SEED_PROMPT = "colorful photorealistic urban scene"  # must match training

tokeniser = pipe.tokenizer
text_encoder = pipe.text_encoder

with torch.no_grad():
    tokens = tokeniser(
        SEED_PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=tokeniser.model_max_length,
        truncation=True
    ).to(device)
    seed_emb = text_encoder(**tokens).last_hidden_state  # [1, 77, 768]


# ---- Load and apply delta ----
delta = torch.load("best_delta.pt", map_location=device)  # [1, 77, 768]
print(f"delta shape: {delta.shape}, seed_emb shape: {seed_emb.shape}")
assert delta.shape == seed_emb.shape, "Shape mismatch — check how delta was saved"

prompt_embeds = (seed_emb + delta).to(dtype)  # [1, 77, 768]

"""
pipe.load_textual_inversion("best_delta.pt", token="S*")
prompt_embeds = SEED_PROMPT + " S*"  
"""
# ---- Null embedding for CFG negative ----
with torch.no_grad():
    null_tokens = tokeniser(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=tokeniser.model_max_length,
        truncation=True
    ).to(device)
    negative_prompt_embeds = text_encoder(**null_tokens).last_hidden_state.to(dtype)

# ---- Load edge image ----
edge_path = "outputs/edges_hed_custom/edges_hedI00323.png"
edge_img = load_image(edge_path).resize((512, 512))

# ---- Run pipeline ----
output = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    image=edge_img,
    num_inference_steps=24,
    controlnet_conditioning_scale=1.0,
    guidance_scale=6.0
).images[0]

output.save("outputs/delta_result_4.png")
print("Saved to outputs/delta_result_4.png")
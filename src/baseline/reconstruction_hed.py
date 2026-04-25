from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import torch
from diffusers.utils import load_image


def init():
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
    
    return pipe, device, dtype


def generate(pipe, device, dtype, input_path, output_path):
    # ---- Reconstruct seed_emb exactly as in training ----
    SEED_PROMPT = "photorealistic urban scene, high resolution, 8k, sharp, structured"  # must match training

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
    delta = torch.load("best_thermal_token.pt", map_location=device)  # [1, 77, 768]
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
    edge_path = input_path
    edge_img = load_image(edge_path).resize((512, 512))

    # ---- Run pipeline ----
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=edge_img,
        num_inference_steps=34,
        controlnet_conditioning_scale=1.0,
        guidance_scale=6.0
    ).images[0]

    output.save(output_path)
    print(f"Saved to {output_path}")
    
if __name__ == "__main__":
    pipe, device, dtype = init()
    delta = torch.load("best_thermal_token.pt", map_location=device) 
    print ("shape of delta:", delta.shape)
    print (torch.norm(delta).item())
    generate(
        pipe=pipe,
        device=device,
        dtype=dtype,
        input_path="debug/ep5_set01_V005_I00150_fused.png",
        output_path="outputs/reconstruction_hed/recon_thermal_v3.png"
    )
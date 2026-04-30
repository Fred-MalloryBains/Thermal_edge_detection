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


def load_textual_inversion(pipe,device, delta_path):
    token_name = "T*"
    
    delta =  torch.load(delta_path, map_location=device)  # [1, 77, 768]
    num_added = pipe.tokenizer.add_tokens(token_name)
    assert num_added == 1, "Token already exists"
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    
    token_id = pipe.tokenizer.convert_tokens_to_ids(token_name)
    with torch.no_grad():
        token_embeddings = pipe.text_encoder.get_input_embeddings()
        token_embeddings.weight[token_id] = delta.to(pipe.text_encoder.device)
        
    return pipe, token_name
    

def generate(pipe, device, dtype, token_name, input_path, output_path):
    # ---- Reconstruct seed_emb exactly as in training ----

    SEED_PROMPT_NT = "photo of an urban car park scene, high resolution, 8k, sharp, structured" # must match training
    SEED_PROMPT = "colour photo"
    NEGATIVE_PROMPT_NT = """
    blurry, low quality, deformed, cartoon, painting, illustration,
    oversaturated, monochrome, distorted geometry
    """
    
    
    # ---- Load edge image ----
    edge_path = input_path
    edge_img = load_image(edge_path).resize((512, 512))

    # ---- Run pipeline ----
    output = pipe(
        prompt=SEED_PROMPT + f" {token_name}",  # Use the new token in the prompt
        negative_prompt=negative_prompt,
        image=edge_img,
        num_inference_steps=34,
        controlnet_conditioning_scale=1.0,
        guidance_scale=6.0
    ).images[0]

    output.save(output_path)
    print(f"Saved to {output_path}")
    
if __name__ == "__main__":
    pipe, device, dtype = init()

    pipe, token_names = load_textual_inversion(pipe, device, "weights/best_KAIST_tokens_big.pt")

    generate(
        pipe=pipe,
        device=device,
        dtype=dtype,
        token_names=None,
        input_path="outputs/final_comparison/edges/thermal_I00000.png",
        #input_path = "debug/ep0_set01_V005_I00150_fused.png",
        output_path="outputs/final_comparison/recon/I00000.png"
    )
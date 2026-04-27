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

    token_names = ["<KAIST-1>", "<KAIST-2>", "<KAIST-3>"]
    delta_list = torch.load(delta_path, map_location=device) # [1, 77, 768]



    pipe.tokenizer.add_tokens(token_names)
    # Resize the embedding layer once
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))


    with torch.no_grad():
        token_embeddings = pipe.text_encoder.get_input_embeddings()

        for token_name, delta in zip(token_names, delta_list):

            token_id = pipe.tokenizer.convert_tokens_to_ids(token_name)

            # Ensure the embedding is on the correct device/dtype
            delta_tensor = delta.to(pipe.text_encoder.device)

            # Assign the weight

            token_embeddings.weight[token_id] = delta_tensor

            print(f"Loaded {token_name} at ID {token_id}")


    return pipe, token_names




def generate(pipe, device, dtype, input_path, output_path, token_names=None):

    # ---- Reconstruct seed_emb exactly as in training ----

    SEED_PROMPT_NT = "photorealistic urban scene, high resolution, 8k, sharp, structured" # must match training
    SEED_PROMPT = "colour photo"
    NEGATIVE_PROMPT_NT = """
    blurry, low quality, deformed, cartoon, painting, illustration,
    oversaturated, monochrome, distorted geometry
    """
    NEGATIVE_PROMPT = ""

    if token_names:
        token_string = " ".join(token_names)
        prompt = f"{token_string} {SEED_PROMPT_NT}"
        negative_prompt = NEGATIVE_PROMPT_NT
    else:
        prompt = SEED_PROMPT_NT
        negative_prompt = NEGATIVE_PROMPT_NT

    # ---- Load edge image ----

    edge_path = input_path

    edge_img = load_image(edge_path).resize((512, 512))

    # ---- Run pipeline ----

    output = pipe(

        prompt=prompt, # Use the new token in the prompt
        negative_prompt=negative_prompt,
        image=edge_img,
        num_inference_steps=34,
        controlnet_conditioning_scale=1.0,
        guidance_scale=5.0
    ).images[0]

    output.save(output_path)

    print(f"Saved to {output_path}")

if __name__ == "__main__":

    pipe, device, dtype = init()

    pipe, token_names = load_textual_inversion(pipe, device, "weights/best_KAIST_tokens.pt")

    generate(
        pipe=pipe,
        device=device,
        dtype=dtype,
        token_names=token_names,
        input_path="debug/ep60_set01_V005_I00150_fused.png",
        output_path="outputs/reconstruction_hed/recon_thermal_v9.png"
    )
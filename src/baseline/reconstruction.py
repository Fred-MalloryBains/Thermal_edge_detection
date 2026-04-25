from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import numpy as np
import torch
from pathlib import Path 
import numpy as np 
from diffusers.utils import load_image



def generate(input_path, output_path, prompt, negative_prompt, device, dtype):
    # Control Net model
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed", 
            torch_dtype=dtype
        )
    except Exception as error:
        print (error)

    print ('load controlnet')

    # --------- LOAD MAIN PIPELINE ----------
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        torch_dtype=dtype
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    
    
    edge_img = load_image(input_path).resize((512, 512))
    
    # Run inference
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_img,
        num_inference_steps=34, # UniPC is fast; 20-25 steps is usually enough
        controlnet_conditioning_scale=1.0,
        guidance_scale=9
    ).images[0]

    output.save(output_path)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    prompt = """
    photorealistic, urban scene, high resolution, 8k, sharp, structured
    """

    negative_prompt = """
    blurry, low quality, deformed, cartoon, painting, illustration,
    oversaturated, monochrome, distorted geometry
    """
    #edge_path = "outputs/baseline/edges/edges_visible_hed.png"
    edge_path = "debug/ep0_set01_V005_I00150_fused.png"
    
    
    generate(
        input_path=edge_path, 
        output_path="outputs/baseline/reconstruction/recon_thermal.png", 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        device=device,
        dtype=dtype
    )
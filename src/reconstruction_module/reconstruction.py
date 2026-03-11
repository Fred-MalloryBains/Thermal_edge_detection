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


#edge_image = load_image("~/downloads/data/I00000.jpg")


# Control Net model
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float32
    )
except Exception as error:
    print (error)

print ('load controlnet')

# --------- LOAD MAIN PIPELINE ----------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float32
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)



prompt = """
photorealistic, realistic image, high resolution, ultra detailed,
accurate textures, natural lighting, soft shadows,
DSLR photograph, sharp focus, cinematic lighting, 8k, high dynamic range
"""

negative_prompt = """
blurry, low quality, deformed, cartoon, painting, illustration,
oversaturated, monochrome, distorted geometry, unrealistic lighting
"""

for i in range (0,10):
    edge_image = load_image(f"outputs/edge_maps/edges_{i}.png")
    edge_image.resize((512,512))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_image,
        num_inference_steps=34,   # reduced for speed
        controlnet_conditioning_scale=1.0,
        guidance_scale=7.5
    )

    # --------- SAVE RESULT ----------
    out.images[0].save(f"outputs/results/result{i}.jpg")

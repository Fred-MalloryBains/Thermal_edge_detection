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


device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# Control Net model
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
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



prompt = """
photorealistic, realistic colours, high resolution, 8k, sharp, structured
"""

negative_prompt = """
blurry, low quality, deformed, cartoon, painting, illustration,
oversaturated, monochrome, distorted geometry
"""

for i in range(3):
    # Process both thermal and visible
    for mode in ["thermal", "visible"]:
        edge_path = f"outputs/edges/edges_{mode}_{i}.png"
        
        # FIX: Re-assign the resized image
        edge_img = load_image(edge_path).resize((512, 512))
        
        # Run inference
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=edge_img,
            num_inference_steps=34, # UniPC is fast; 20-25 steps is usually enough
            controlnet_conditioning_scale=1.0,
            guidance_scale=9
        ).images[0]

        output.save(f"outputs/results/{mode}_result{i}.jpg")

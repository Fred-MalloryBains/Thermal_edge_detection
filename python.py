from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
from scipy import ndimage
import cv2

data_path = Path("/Volumes/Pluggable_1TB/thermal_images/archive/").expanduser()


thermal_paths = list(data_path.glob("set*/V*/lwir"))


all_files = []

for lwir_dir in thermal_paths:
    if lwir_dir.is_dir():
        files = list(lwir_dir.iterdir()) 
        all_files.extend(files)



print ('import')


def sobel_edge_detector(image_path, flag=True):
    image_path = str(image_path)
    if flag:
        image = cv2.imread(image_path, 0)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold = 50
    edges = (magnitude > threshold).astype(np.uint8) * 255
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges

def roberts_cross(img_path):
    img_path = str(img_path)
    # Read the image using OpenCV
    #Reads the image from the specified path in grayscale, simplifying processing.
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Define the Roberts Cross kernels for horizontal and vertical edge detection.
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])
    
    # Apply convolution operations to calculate horizontal and vertical gradients.
    gradient_x = ndimage.convolve(image, gx)
    gradient_y = ndimage.convolve(image, gy)

    # Compute the magnitude of the gradient using the square root of the sum of squares.
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Apply a threshold to create a binary mask, where pixels with magnitudes exceeding the threshold are considered part of an edge.
    threshold = 10
    edges = (magnitude > threshold).astype(np.uint8) * 255
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges

# edge detection as rgb
#edges_img_sobel = [sobel_edge_detector(all_files[x]) for x in range(2)]
#edges_img_roberts = [roberts_cross(all_files[x]) for x in range(2)]
print (all_files[0])
edges_img = cv2.Canny(cv2.imread(str(all_files[0])),10,70)
# Must save BEFORE loading into ControlNet
Image.fromarray(edges_img).save("canny_edges.png")
edge_image = load_image("canny_edges.png")

print ('load image')

# Control Net model
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16
    )
except Exception as error:
    print (error)

print ('load controlnet')

# --------- LOAD MAIN PIPELINE ----------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


# --------- INFERENCE ----------
prompt = "a nightime urban scene with buildings trees \
with realistic RGB reconstruction, \
natural color palette, correct illumination, physically accurate shading, \
DSLR photo, high detail, 35mm lens, ultra sharp, realistic textures, \
accurate structures, true-to-life colors, environment consistency, no thermal artifacts"


negative_prompt = "blurry, glowing artifacts, thermal noise, distorted colors, \
incorrect illumination, unrealistic textures, over-saturation, \
infrared look, thermal colors, false color mapping"


out = pipe(
    prompt=prompt,
    guess_mode=True,
    negative_prompt=negative_prompt,
    image=edge_image,
    num_inference_steps=30,   # reduced for speed
)

# --------- SAVE RESULT ----------
out.images[0].save("reconstructed.png")
print("reconstructed.png")
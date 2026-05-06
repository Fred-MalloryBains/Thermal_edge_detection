"""
Evaluation pipeline is the full implementaiton for handling conversion from thermal imaging to  
RGB using ControlNet.
  
This code uses arguments to specify the intended input and output directories of the images with 
default values matching the repository structure

Further conditioning to be added TBD
"""

#import tools.dataloader import ThermalDataset
import argparse 
from PIL import Image
import torch
import os

from src.preprocess.edge_detector_custom_hed import init as custom_hed_init
from src.preprocess.edge_detector_custom_hed import process_edge_pytorch
from src.reconstruction_module.token_reconstruction import generate, init_reconstruction, load_textual_inversion
from src.baseline.edge_detector_hed import hed_init as baseline_hed_init
from src.baseline.edge_detector_hed import process_edge_thermal

EDGE_DIRECTORY = "outputs/edges"

def create_edge_map(input_file):
    # Try custom HED first, if it fails, fall back to baseline HED
    try:
        if os.path.exists("weights/hed_thermal.pth"):
            print("Loading custom HED model from weights/thermal_hed.pth")
            device, model = custom_hed_init()
            edge_map = process_edge_pytorch(input_file, model, device)
            edge_map = Image.fromarray(edge_map)
        else:
            print("custom HED model not found, using baseline HED model")
            device, model = baseline_hed_init()
            edge_map = process_edge_thermal(input_file, model, device)
            edge_map = Image.fromarray(edge_map)
        edge_map.save(EDGE_DIRECTORY + "/" + os.path.basename(input_file).replace(".jpg", "_edge.png"))
    except Exception as error:
        print(f"Error during edge map creation: {error}")
        exit(1)
    return edge_map

def reconstruct_image(edge_map, output_path):
    # Try to reconstruct the image using the learned tokens, if it fails, run without tokens
    try:
        pipe, device, dtype = init_reconstruction()
    except Exception as error:
        print(f"Error during pipeline initialization: {error}")
        exit(1)
    try:
        if not os.path.exists("weights/best_KAIST_tokens.pt"):
            print("Textual inversion weights not found at weights/best_KAIST_tokens.pt")
            generate(
                pipe = pipe,
                device=device,
                dtype=dtype, 
                input_path=edge_map, 
                output_path=output_path,
            )
        else:
            print("Loading textual inversion weights from weights/best_KAIST_tokens.pt")
            pipe, token_names = load_textual_inversion(pipe, device, "weights/best_KAIST_tokens.pt")
            generate(
                pipe = pipe,
                device=device,
                dtype=dtype, 
                input_path=edge_map, 
                output_path=output_path,
                token_names=token_names
            )
    except Exception as error:
        print(f"Error during reconstruction: {error}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline for thermal to RGB conversion")
    parser.add_argument("--input_file", type=str, default="outputs/baseline/lwir/example_one", help="Directory containing edge maps")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline/reconstruction/", help="Directory to save reconstructed images")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(EDGE_DIRECTORY):
        os.makedirs(EDGE_DIRECTORY)
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        exit(1)
    
    # Process the input file to create an edge map
    print(f"Processing input file: {args.input_file}")
    edge_map = create_edge_map(args.input_file)
    output_path = os.path.join(args.output_dir,
                               os.path.basename(args.input_file).replace(".jpg", "_recon.png")
                               )
    print(f"Saving reconstructed image to: {output_path}")
    
    # Reconstruct the image from the edge map
    reconstruct_image(edge_map, output_path)
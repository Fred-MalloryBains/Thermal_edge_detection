"""
Evaluation pipeline is the full implementaiton for handling conversion from thermal imaging to  
RGB using ControlNet.
  
This code uses arguments to specify the intended input and output directories of the images with 
default values matching the repository structure

Further conditioning to be added TBD
"""

from tools.dataloader import ThermalDataset
import argparse 
import Image
import torch
import os

from src.preprocess.edge_detector_custom_hed import process_edge_pytorch
from baseline.reconstruction_hed import generate, init

EDGE_DIRECTORY = "outputs/edges"

def create_edge_map(input_dir): 
    img = Image.open(input_dir).convert("RGB")
    edge_map = process_edge_pytorch(img)
    edge_map = Image.fromarray(edge_map)
    edge_map.save(EDGE_DIRECTORY + "/" + os.path.basename(input_dir).replace(".jpg", "_edge.png"))
    return edge_map

def reconstruct_image(edge_map, output_path):
    try:
        pipe, device, dtype = init()
    except Exception as error:
        print(f"Error during pipeline initialization: {error}")
        exit(1)
    try:
        generate(
            pipe = pipe,
            device=device,
            dtype=dtype, 
            input_path=edge_map, 
            output_path=output_path,
        )
    except Exception as error:
        print(f"Error during reconstruction: {error}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline for thermal to RGB conversion")
    parser.add_argument("--input_file", type=str, default="outputs/baseline/edges/", help="Directory containing edge maps")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline/reconstruction/", help="Directory to save reconstructed images")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(EDGE_DIRECTORY):
        os.makedirs(EDGE_DIRECTORY)
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        exit(1)
    
    edge_map = create_edge_map(args.input_file)
    output_path = os.path.join(args.output_dir,
                               os.path.basename(args.input_file).replace(".jpg", "_recon.png")
                               )
    reconstruct_image(edge_map, output_path)
    
    
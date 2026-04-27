# Thermal image reconstruction pipeline

This code follows an investigation for bridging the domain gap between VL and IR images using 
ControlNet, a conditioning constraint for stable diffusion. This project aims to improve the interpretability of thermal images through improved performance metrics centred around practical applications such as low light surveilance and autonomous vehicles.

## Implementation

To run the end-to-end thermal to reconstruction pipeline, clone the repository and create a virtual environment to install the dependencies in the `requirements.txt` file:

```
python3.11 -m venv .venv
pip install -r requirements.txt
```

To pipeline is handled in `tools/'evaluation_pipe.py`, call this in the command line with the arguments as shown:

```
python -m tools.evaluation.pipe --input_file <path_to_image_file> --output_dir <path_to_output_directory>
```

Results are save as a .png file in the directory specified, example usage shown below: 

[placeholder for image]


## Training setup 

The image translation pipeline makes use of a deep learning edge detection network HED to condition stable diffusion through controlNet

[HED placeholder]

[ControlNet placeholder]

The edge detection was tuned on refining the thermal outputs to reduce focal, boundaryIoU and DICE loss. This implementation is found in `src/preprocess/train_hed_thermal` with the data loaded from `tools/dataloader.py`
The saved model weights named `hed_thermal.pth`

[before transfer learning]
[after transfer learning]

To capture the scene reconstruction through the stable diffusion prompt, textual inversion was trained using `src/reconstruction_module/text_inversion.py`. 
The saved tokens are named `best_thermal_token.pt`


## Key references 

ControlNet: 
Textual Inversion:
Hollistically Nested Edge Detection: 
Base HED model: 

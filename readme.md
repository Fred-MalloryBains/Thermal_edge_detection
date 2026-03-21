# Thermal image reconstruction pipeline

This code follows an investigation for bridging the domain gap between VL and IR images using 
ControlNet, a conditioning constraint for stable diffusion. This project aims to improve the interpretability of thermal images through improved performance metrics centred around practical applications such as low light surveilance and autonomous vehicles.

## Edge Detection

ControlNet uses image-to-image translation, which relies on an edge map image input generated from the thermal image. In this project, HED (Hollistically Nested Edge detection) is used to generate the edge mappings. The preprocessing module uses openCV and a 2x2 CNN bridges the domain gap between the "blobs", the inputs for the DNN network, to construct an edge mapping.

Logic for this is found in [/src/preprocess/edge_detector_hed.py](/src/preprocess/edge_detector_hed.py) and is saved in [/outputs/edged_hed](/outputs/edges_hed)

Eg:
![HED edge detection example](/outputs/edges_hed/edges_hed_example.png)

<br>
## Reconsturction uses controlNet

ControlNet reconstruction uses a pretrained stable diffusion model and edge detection to tranlsate the image into the VL domain. The prompt has been designed using a loss model for tokenisation using the evaluation metrics discussed below.
a
The script for this is found in [/src/reconstruction_module/reconstruction_hed.py](/src/reconstruction_module/reconstruction_hed.py) and is saved in [/outputs/results](/outputs/results)
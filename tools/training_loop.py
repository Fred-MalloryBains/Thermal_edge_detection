"""
This code module handles the training loop for the entire pipeline for reconstructing thermal images
into RGB outuputs

There are two components HED network training and textual (prompt) inversion

The argument parser can handle configuring them separately or together, as well as the specified 
location of the dataset. 
"""

import argparser 
from tools.dataloader.py import EdgeToImageDataset
from src.preprocess.train_hed_thermal import train

class trainer: 
    def __init__(self, data_location, train_hed=True, train_text = True):
        self.dataset = EdgeToImageDataset(self.data_location)
        if train_hed:
            self.run_hed()
        elif train_text:
            self.run_train_text()
    def run_train_text(self):
        
        train(self.dataset)
        
    def run_hed(self):
        pass


if __name__ == "__main__":
    aguments = argparser
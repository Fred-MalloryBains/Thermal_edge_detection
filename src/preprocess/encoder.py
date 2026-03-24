import torch 
import torch.nn as nn

class ThermalEncoder(nn.Module):
    def __init__(self):
        super(ThermalEncoder, self).__init__()
        self.feature_extractor = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        
        self.domain_mapper = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)  # Maps to 3 channels for HED compatibility
        
    def forward(self, x):
        x = self.relu(self.feature_extractor(x))
        x = self.domain_mapper(x)
        
        return torch.sigmoid(x)  # normalise
    
    

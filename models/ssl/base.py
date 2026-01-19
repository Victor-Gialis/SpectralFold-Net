import torch 
import torch.nn as nn

class BaseSSLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")
    
    def compute_loss(self, predictions, targets):
        raise NotImplementedError("Compute_loss method not implemented!")
    
    def non_negative_output(self, x):
        return torch.clamp(x, min=0.0)

    @classmethod
    def from_config(cls, config:dict):
        raise NotImplementedError("From Config method not implemented")
import torch

class GaussianNoise:
    
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size())*self.std + self.mean
        return tensor + noise
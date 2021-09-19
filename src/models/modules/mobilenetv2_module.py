"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from torch import nn 
import torch 

class Mobilenetv2Module(nn.Module):
    """
    """
    def __init__(self, hparams:dict):
        """
        """
        super().__init__()

    def forward(self, x:torch.Tensor):
        return x
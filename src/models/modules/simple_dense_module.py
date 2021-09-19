"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from torch import nn
import torch 

class SimpleDenseModule(nn.Module):
    """This class represents the simple dense net 
    """
    def __init__(self, hparams:dict):
        """
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            nn.BatchNorm1d(hparams["lin1_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            nn.BatchNorm1d(hparams["lin2_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            nn.BatchNorm1d(hparams["lin3_size"]),
            nn.ReLU(),
            nn.Linear(hparams["lin3_size"], hparams["output_size"])
        )

    def forward(self, x:torch.Tensor):
        batch_size, channels, width, height = x.size()
        # Convert (batch_size, channels, width, height) -> (batch_size, channels*width*height)
        x = x.view(batch_size, -1)
        return self.model(x)

    
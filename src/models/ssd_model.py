"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from .modules import Mobilenetv2Module
from pytorch_lightning import LightningModule
import torch 
from torch.nn import functional as F
from typing import Any

class SSDModel(LightningModule):
    """
    """
    def __init__(self,
            learning_rate: float=0.001,
            weight_decay: float=0.0005
        ):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Backbone 
        self.model = Mobilenetv2Module(self.hparams)

    def forward(self, x:torch.Tensor):
        """
        """
        return self.model(x)

    def criterion(self, logits:torch.Tensor, y:torch.Tensor):
        """
        """
        return F.cross_entropy(logits, y)

    def step(self, batch:Any):
        """
        """
        x, y = batch 
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss, preds, y 

    def on_epoch_start(self):
        """
        """
        print("\n")

    def training_step(self, batch:Any, batch_idx:int):
        """
        """
        loss, preds, y = self.step(batch)

        # Logging 
        self.log("Train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "y":y}

    def validation_step(self, batch:Any, batch_idx:int):
        """
        """
        loss, preds, y = self.step(batch)

        # Logging 
        self.log("Val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "y":y}

    def test_step(self, batch:Any, batch_idx:int):
        """
        """
        loss, preds, y = self.step(batch)

        # Loggin 
        self.log("Test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "y":y}

    def configure_optimizers(self):
        """
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        ) 

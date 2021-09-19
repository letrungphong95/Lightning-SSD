"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from pytorch_lightning import LightningModule
from .modules import SimpleDenseModule
import torch
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy
from typing import Any, List

class MNISTModel(LightningModule):
    """This class represents the MNIST Classification model
    """
    def __init__(self, input_size:int=28*28, lin1_size:int=256,
            lin2_size:int=256, lin3_size:int=256, output_size:int=10,
            learning_rate:float=0.001, weight_decay:float=0.0005
        ):
        """Section 1: Computations (init)
        """
        super().__init__()
        self.save_hyperparameters()

        # Backbone 
        self.model = SimpleDenseNet(self.hparams)

        # Accuracy 
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def criterion(self, logits:torch.Tensor, y:torch.Tensor):
        """
        """
        return F.cross_entropy(logits, y)

    def forward(self, x:torch.Tensor):
        """
        """
        return self.model(x)

    def step(self, batch:Any):
        """
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def on_epoch_start(self):
        print('\n')

    def training_step(self, batch:Any, batch_idx:int):
        """Section 2: Train loop (training_step)
        """
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)

        # logging 
        self.log("Train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds":preds, "targets":targets, "acc": acc}

    def training_epoch_end(self, output:List[Any]):
        """ `outputs` is a list of dicts returned from `training_step()`
        """
        pass 

    def validation_step(self, batch:Any, batch_idx:int):
        """Section 3: Validation loop (validation_step)
        """
        loss, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)

        # Logging 
        self.log("Val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Val/acc", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss":loss, "preds":preds, "targets":targets, "acc":acc}

    def validation_epoch_end(self, output:List[Any]):
        """
        """
        pass 

    def test_step(self, batch:Any, batch_ids:int):
        """
        """
        loss, preds, targets = self.step(batch)
        acc = self.test_accuracy(preds, targets)

        # Logging 
        self.log("Test/loss", loss, on_step=False, on_epoch=True)
        self.log("Test/acc", acc, on_step=False, on_epoch=True)

        return {"loss":loss, "preds":preds, "targets":targets, "acc":acc}

    def test_epoch_end(self, output:List[Any]):
        """
        """
        pass 

    def configure_optimizers(self):
        """See examples here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )

        

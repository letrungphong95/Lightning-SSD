"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from typing import Optional

class VOCDataModule(LightningDataModule):
    """
    """
    def __init__(self,
            data_dir: str="data/voc"
        ):
        """
        """
        super().__init__()
        self.save_hyperparameters()
        

    def prepare_data(self):
        """
        """
        pass 

    def setup(self, stage:Optional[str]):
        """
        """
        pass 

    def train_dataloader(self):
        """
        """
        pass 

    def val_dataloader(self):
        """
        """
        pass 

    def test_dataloader(self):
        """
        """
        pass 


    
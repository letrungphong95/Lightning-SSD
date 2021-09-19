"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from pytorch_lightning import LightningDataModule 
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from typing import Tuple, Optional

class MNISTDataModule(LightningDataModule):
    """
    """
    def __init__(self,
            data_dir: str='data/',
            dataset_split: Tuple[float, float, float]=(55_000, 5_000, 10_000),
            batch_size: int=32,
            num_workers: int=0,
            pin_memory: bool=False
        ):
        super().__init__()
        self.save_hyperparameters()

        # Transform 
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        # MNIST Dataset object 
        self.data_train: Optional[Dataset] = None 
        self.data_val: Optional[Dataset] = None 
        self.data_test: Optional[Dataset] = None 
    
    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed 
        """
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)


    def setup(self, stage:Optional[str]):
        """Load dataset varaiable 
        """
        trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
        dataset = ConcatDataset(datasets=[trainset, testset])
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.hparams.dataset_split
        )
    
    def train_dataloader(self):
        """
        """
        return DataLoader(
            dataset = self.data_train,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True
        )

    def val_dataloader(self):
        """
        """
        return DataLoader(
            dataset = self.data_val,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers,
            pin_memory = self.hparams.pin_memory, 
            shuffle = False
        )

    def test_dataloader(self):
        """
        """
        return DataLoader(
            dataset = self.data_test,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers, 
            pin_memory = self.hparams.pin_memory,
            shuffle = False
        )


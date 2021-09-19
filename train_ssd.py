"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from train_mnist import AVAIL_GPUS
from pytorch_lightning import Trainer 
from argparse import ArgumentParser
from src.config import Config 
from src.models import SSDModel
from src.datamodules import VOCDataModule 
from src.callbacks import checkpoint_callback 
import torch 
AVAIL_GPUS = min(1, torch.cuda.device_count())

if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str)
    args = parser.parse_args()
    config = Config(args.config_file)

    if config.dataset == 'ssd':
        # ------------
        # data
        # ------------
        data_loader = VOCDataModule(
            data_dir=config.data_dir
        )

        # ------------
        # model
        # ------------
        model = SSDModel(
            learning_rate=config.learning_rate, 
            weight_decay=config.weight_decay
        )

        # ------------
        # training
        # ------------
        trainer = Trainer(
            max_epochs=config.num_epoch,
            gpus=AVAIL_GPUS,
            progress_bar_refresh_rate=20,
            callbacks=[checkpoint_callback]
        )
        trainer.fit(model, data_loader)

        # ------------
        # testing
        # ------------
        result = trainer.test(datamodule=data_loader)



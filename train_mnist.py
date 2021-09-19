"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser 
from src.config import Config
from src.models import MNISTModel, SSDModel
from src.datamodules import MNISTDataModule, VOCDataModule
from src.callbacks import checkpoint_callback
import torch
AVAIL_GPUS = min(1, torch.cuda.device_count())

if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str)
    args = parser.parse_args()
    config = Config(args.config_file)

    if config.dataset == 'mnist':
        # ------------
        # data
        # ------------
        data_loader = MNISTDataModule(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )

        # ------------
        # model
        # ------------
        model = MNISTModel(
            input_size=config.size[1]*config.size[2], 
            lin1_size=config.layers[0],
            lin2_size=config.layers[1], 
            lin3_size=config.layers[2], 
            output_size=config.num_classes,
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



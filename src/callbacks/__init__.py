"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from pytorch_lightning.callbacks import ModelCheckpoint

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor="Val/loss",
    dirpath="model",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=5,
    mode="min",
)
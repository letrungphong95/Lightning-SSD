"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from src.datamodules import MNISTDataModule
from src.config import Config

config = Config('config/base_model.yaml')
# print('dkm', config, **cfg)
dataset = MNISTDataModule(
    data_dir=config.data_dir,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory
)
dataset.prepare_data()
dataset.setup()

print(len(dataset.data_train))
# use_data 
assert len(dataset.data_train) == 55000
for i, batch in enumerate(dataset.train_dataloader()):
    assert batch[0].size() == (config.batch_size, 1, 28, 28)
    assert batch[1].size() == (config.batch_size, )
    if i==10:
        break 

assert len(dataset.data_val) == 5000
for i, batch in enumerate(dataset.val_dataloader()):
    assert batch[0].size() == (config.batch_size, 1, 28, 28)
    assert batch[1].size() == (config.batch_size, )
    if i==10:
        break 

assert len(dataset.data_test) == 10000
for i, batch in enumerate(dataset.test_dataloader()):
    assert batch[0].size() == (config.batch_size, 1, 28, 28)
    assert batch[1].size() == (config.batch_size, )
    if i==10:
        break 


# # download, etc...
# dataset = VOCDataset()
# dataset.prepare_data()

# # splits/transforms
# dataset.setup(stage="fit")

# # use data
# for batch in dataset.train_dataloader():
#     pass 
# for batch in dataset.val_dataloader():
#     pass 

# dataset.teardown(stage="fit")

# # lazy load test data
# dataset.setup(stage="test")
# for batch in dataset.test_dataloader():
#     pass 

# dataset.teardown(stage="test")


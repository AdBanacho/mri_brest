from functools import partial

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .NiftiDataset import NiftiDataset
from .pad_collate import pad_collate
from mriBreastDuke.constants import NIFTI_PATH

from mriBreastDuke.dataLoaders.subtraction import SUBTRACTION_NONE, get_input_channels

class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, target_size=None, image_root=NIFTI_PATH, batch_size=2, num_workers=4, subtraction_mode=SUBTRACTION_NONE):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.image_root = image_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.subtraction_mode = subtraction_mode
        self.input_channels = get_input_channels(subtraction_mode)

    def setup(self, stage=None):
        self.train_ds = self.setup_dataset(self.train_df, "train")
        self.val_ds = self.setup_dataset(self.val_df, "val")

    def setup_dataset(self, dataset, label):
        return NiftiDataset(
            dataset,
            size_cache_path=label,
            target_size=self.target_size,
            image_root=self.image_root,
            use_monai=True,
            subtraction_mode=self.subtraction_mode
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(pad_collate, max_series=self.input_channels),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(pad_collate, max_series=self.input_channels),
        )

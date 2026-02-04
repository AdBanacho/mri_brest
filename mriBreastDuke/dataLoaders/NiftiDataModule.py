from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .NiftiDataset import NiftiDataset
from .pad_collate import pad_collate
from .BucketBySizeSampler import BucketBySizeSampler
from mriBreastDuke.constants import NIFTI_PATH, SEED


class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, df, grouped_by_study, target_size=None, image_root=NIFTI_PATH, batch_size=2, num_workers=4):
        super().__init__()
        self.df = df
        self.grouped_by_study = grouped_by_study
        self.image_root = image_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size

    def setup(self, stage=None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=SEED,
            stratify=self.df["label"],
        )

        self.train_ds = self.setup_dataset(train_df, 'train')
        self.val_ds = self.setup_dataset(val_df, 'val')

    def setup_dataset(self, dataset, label):
        return NiftiDataset(dataset,
                            grouped_by_study=self.grouped_by_study,
                            size_cache_path=label,
                            target_size=self.target_size,
                            image_root=self.image_root,
                            use_monai=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            # batch_sampler=BucketBySizeSampler(self.train_ds, batch_size=self.batch_size),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            # batch_sampler=BucketBySizeSampler(self.val_ds, batch_size=self.batch_size),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate
        )

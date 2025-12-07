import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from monai.networks.nets import DenseNet121

from readFromXlmx import get_oncotype_score_for_series_as_serie_and_label_df
from load_or_compute_sizes import load_or_compute_sizes

BASE_PATH = 'mri_breast_duke/'
NIFTI_PATH = "images/tciaNifti"

SEED = 42


def pad_collate(batch):
    """
    batch: list of (vol, label)
      vol: (1, D, H, W)
    Returns:
      vols: (B, 1, Dmax, Hmax, Wmax)
      labels: (B,)
    """
    vols, labels = zip(*batch)  # list of tensors, list of tensors

    # Ensure shape is (1, D, H, W)
    fixed_vols = []
    for v in vols:
        v = v.float()
        if v.ndim == 3:
            v = v.unsqueeze(0)          # (D, H, W) -> (1, D, H, W)
        assert v.ndim == 4, f"Expected 4D tensor, got {v.shape}"
        assert v.shape[0] == 1, f"Expected 1 channel, got {v.shape}"
        fixed_vols.append(v)

    # Compute max spatial dims in this batch
    max_d = max(v.shape[1] for v in fixed_vols)
    max_h = max(v.shape[2] for v in fixed_vols)
    max_w = max(v.shape[3] for v in fixed_vols)

    padded_vols = []
    for v in fixed_vols:
        _, d, h, w = v.shape
        pad_d = max_d - d
        pad_h = max_h - h
        pad_w = max_w - w

        # F.pad: (w_left, w_right, h_left, h_right, d_left, d_right)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        v_padded = F.pad(v, padding, mode="constant", value=0.0)
        padded_vols.append(v_padded)

    # Stack along batch dimension → (B, 1, Dmax, Hmax, Wmax)
    vols_tensor = torch.stack(padded_vols, dim=0)

    labels_tensor = torch.stack(labels, dim=0)  # (B,)
    return vols_tensor, labels_tensor


class BucketBySizeSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Convert shape into a single scalar: volume = D*H*W
        self.keys = [np.prod(s) for s in dataset.sizes]

        # Sort indices by size
        self.indices = np.argsort(self.keys)

    def __iter__(self):
        # Yield sorted indices in batches
        batch = []
        for idx in self.indices:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1


class NiftiDataset(Dataset):
    def __init__(self, df,
                 target_size=None,
                 image_root=NIFTI_PATH,
                 target_col="label",
                 serie_col="serie",
                 size_cache_path='train'):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_col = target_col
        self.serie_col = serie_col
        self.target_size = target_size
        self.size_cache_path = os.path.join(BASE_PATH, size_cache_path + "_volume_sizes.csv")
        self.sizes = load_or_compute_sizes(self.df, self.size_cache_path, self.image_root, self.serie_col)

    def __len__(self):
        return len(self.df)

    def _load_nifti(self, serie):
        path = os.path.join(self.image_root, f"{serie}.nii.gz")
        vol = nib.load(path).get_fdata().astype(np.float32)

        vol = np.squeeze(vol)
        vol = self.z_score(vol)
        vol = np.expand_dims(vol, 0)  # (1, D, H, W)
        vol = torch.from_numpy(vol)

        if self.target_size is not None:
            vol = F.interpolate(
                vol.unsqueeze(0),
                size=self.target_size,
                mode="trilinear",
                align_corners=False
            ).squeeze(0)

        return vol

    def z_score(self, vol):
        mean = vol.mean()
        std = vol.std() + 1e-5
        return (vol - mean) / std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        serie = row[self.serie_col]
        label = row[self.target_col]

        vol = self._load_nifti(serie)

        label = torch.tensor(label, dtype=torch.long)

        return vol, label


class Simple3DFCN(nn.Module):
    """
    Fully convolutional 3D network:
    - Conv3d + MaxPool3d blocks
    - 1x1x1 Conv3d to get num_classes channels
    - AdaptiveAvgPool3d(1) to aggregate over D,H,W
    No Linear layers.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        # 1x1x1 conv to map 32 feature channels -> num_classes
        self.classifier_conv = nn.Conv3d(32, num_classes, kernel_size=1)

        # Global adaptive average pooling to get (B, C, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Feature extractor
        x = self.pool(F.relu(self.conv1(x)))  # (B, 8, D/2, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 16, D/4, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 32, D/8, H/8, W/8)

        # Class logits per spatial location
        x = self.classifier_conv(x)           # (B, num_classes, D/8, H/8, W/8)

        # Global average pooling over D,H,W -> (B, num_classes, 1, 1, 1)
        x = self.global_pool(x)

        # Flatten to (B, num_classes)
        x = x.view(x.size(0), -1)
        return x


class DebugBatchShapeCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[TRAIN] batch {batch_idx} shape: {tuple(x.shape)}", flush=True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[VAL]   batch {batch_idx} shape: {tuple(x.shape)}", flush=True)


class NiftiClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        return None

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, df, target_size=None, image_root=NIFTI_PATH, batch_size=2, num_workers=4):
        super().__init__()
        self.df = df
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
        return NiftiDataset(dataset, size_cache_path=label, target_size=self.target_size, image_root=self.image_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_sampler=BucketBySizeSampler(self.train_ds, batch_size=self.batch_size),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_sampler=BucketBySizeSampler(self.val_ds, batch_size=self.batch_size),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate
        )


def train(df, modelName, model):
    data_module = NiftiDataModule(
        df,
        image_root=BASE_PATH + NIFTI_PATH,
        batch_size=4,
        num_workers=4,
    )

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=modelName
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[DebugBatchShapeCallback()],
        log_every_n_steps=1,
        enable_progress_bar=False
    )

    trainer.fit(model=model, datamodule=data_module)

    print("\n=== Final Validation Metrics ===")
    print(f"\n ===      {modelName}      ===")
    metrics = trainer.callback_metrics
    for k, v in metrics.items():
        print(f"{k}: {float(v):.4f}")


def main():
    df = get_oncotype_score_for_series_as_serie_and_label_df(50, 12, SEED)
    # df = get_oncotype_score_for_series_as_serie_and_label_df()

    num_classes = len(set(df.label))
    models = [('FCN',
               NiftiClassifier(Simple3DFCN(num_classes=num_classes))),
              ('DenseNet',
               NiftiClassifier(DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes)))]

    for model in models:
        train(df, *model)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

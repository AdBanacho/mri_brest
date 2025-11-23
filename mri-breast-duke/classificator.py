import os
import numpy as np
import nibabel as nib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from readFromXlmx import get_oncotype_score_for_series_as_serie_and_label_df

NIFTI_PATH = "images/tciaNifti"

class NiftiDataset(Dataset):
    def __init__(self, df, image_root=NIFTI_PATH, target_col="label", serie_col="serie"):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_col = target_col
        self.serie_col = serie_col

    def __len__(self):
        return len(self.df)

    def _load_nifti(self, serie):
        path = os.path.join(self.image_root, f"{serie}.nii.gz")
        img = nib.load(path)
        vol = img.get_fdata()

        vol = np.asarray(vol, dtype=np.float32)

        mean = vol.mean()
        std = vol.std() + 1e-5
        vol = (vol - mean) / std

        # Ensure shape is (C, D, H, W) => (1, D, H, W)
        if vol.ndim == 3:
            vol = np.expand_dims(vol, 0)  # add channel dim
        elif vol.ndim == 4:
            # If already has a channel dim, assume it's first
            pass
        else:
            raise ValueError(f"Unexpected volume shape: {vol.shape}")

        return torch.from_numpy(vol)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        serie = row[self.serie_col]
        label = row[self.target_col]

        vol = self._load_nifti(serie)

        label = torch.tensor(label, dtype=torch.long)

        return vol, label


class Simple3DNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(32 * 8 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NiftiClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=4):
        super().__init__()
        self.model = Simple3DNet(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch
        x = F.interpolate(x, size=(64, 64, 64), mode="trilinear", align_corners=False)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, df, image_root=NIFTI_PATH, batch_size=2, num_workers=4):
        super().__init__()
        self.df = df
        self.image_root = image_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42,
            stratify=self.df["label"],
        )

        self.train_ds = NiftiDataset(train_df, image_root=self.image_root)
        self.val_ds = NiftiDataset(val_df, image_root=self.image_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    df = get_oncotype_score_for_series_as_serie_and_label_df()

    data_module = NiftiDataModule(
        df,
        image_root=NIFTI_PATH,
        batch_size=2,
        num_workers=4,
    )

    model = NiftiClassifier(lr=1e-3, num_classes=4)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
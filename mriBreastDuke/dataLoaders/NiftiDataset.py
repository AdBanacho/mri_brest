import os
import numpy as np
import nibabel as nib
from filelock import FileLock

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ResizeD,
)

from .load_or_compute_sizes import load_or_compute_sizes
from mriBreastDuke.constants import NIFTI_PATH, PREPARED_TO_TRAIN_PATH, SIZE_CACHE_PATH, MAX_SERIES_PER_STUDY


class NiftiDataset(Dataset):
    def __init__(self, df,
                 grouped_by_study,
                 target_size=None,
                 image_root=NIFTI_PATH,
                 target_col="label",
                 serie_col="serie",
                 size_cache_path='train',
                 use_monai=True):
        self.df = df.reset_index(drop=True)
        self.grouped_by_study = grouped_by_study
        self.image_root = image_root
        self.target_col = target_col
        self.serie_col = serie_col
        self.target_size = target_size
        self.size_cache_path = SIZE_CACHE_PATH(size_cache_path)
        # self.sizes = load_or_compute_sizes(self.df, self.size_cache_path, self.image_root, self.serie_col)
        self.use_monai = use_monai

    def __len__(self):
        return len(self.df)

    def _load_nifti(self, serie):
        in_path = os.path.join(self.image_root, f"{serie}.nii.gz")
        os.makedirs(PREPARED_TO_TRAIN_PATH, exist_ok=True)
        out_path = os.path.join(PREPARED_TO_TRAIN_PATH, f"{serie}.nii.gz")

        lock = FileLock(out_path + ".lock")

        if self.use_monai:
            return self._load_nifti_monai(lock, in_path, out_path)
        return self._load_nifti_custom(lock, in_path, out_path)

    def _load_nifti_custom(self, lock, in_path, out_path):

        if os.path.exists(out_path):
            return self._load_from_file(out_path)

        with lock:
            if os.path.exists(out_path):
                return self._load_from_file(out_path)

            vol = self._preprocess_and_save_image(in_path, out_path)
            return vol

    def _preprocess_and_save_image(self, in_path, out_path):
        nii = nib.load(in_path)
        vol = nii.get_fdata().astype(np.float32)
        vol = np.squeeze(vol)
        vol = self._z_score(vol)
        vol = np.expand_dims(vol, 0)  # (1, D, H, W)
        vol = torch.from_numpy(vol)
        if self.target_size is not None:
            vol = self._interpolate(vol)
        vol_np = vol.squeeze(0).cpu().numpy()
        new_nii = nib.Nifti1Image(vol_np, affine=nii.affine)
        nib.save(new_nii, out_path)
        return vol

    def _interpolate(self, vol):
        return F.interpolate(
            vol.unsqueeze(0),
            size=self.target_size,
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

    def _load_from_file(self, out_path):
        vol = nib.load(out_path).get_fdata().astype(np.float32)
        return torch.from_numpy(vol).unsqueeze(0)
    def _z_score(self, vol):
        mean = vol.mean()
        std = vol.std() + 1e-5
        return (vol - mean) / std

    def _load_nifti_monai(self, lock, in_path, out_path):

        if os.path.exists(out_path):
            return self._monai_load_cached(out_path)

        with lock:
            if os.path.exists(out_path):
                return self._monai_load_cached(out_path)

            vol = self._monai_preprocess_and_save(in_path, out_path)
            return vol

    def _monai_load_cached(self, path) -> torch.Tensor:
        x = {"img": path}
        t = Compose([
            LoadImaged(keys="img", image_only=False),
            EnsureChannelFirstd(keys="img"),
        ])
        out = t(x)
        return out["img"].float()

    def _monai_preprocess_and_save(self, in_path, out_path):
        x = {"img": in_path}

        tfms = [
            LoadImaged(keys="img", image_only=False),
            EnsureChannelFirstd(keys="img"),
            NormalizeIntensityd(keys="img", nonzero=False, channel_wise=True),
        ]

        if self.target_size is not None:
            tfms.append(ResizeD(keys="img", spatial_size=self.target_size, mode="trilinear", align_corners=False))

        out = Compose(tfms)(x)

        img = out["img"].float()
        meta = out["img_meta_dict"]

        nib.save(
            nib.Nifti1Image(img.squeeze(0).cpu().numpy(), meta["affine"]),
            out_path
        )

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        studyId = row[self.serie_col]
        label = row[self.target_col]

        series_ids = self.grouped_by_study.get_group(studyId)
        series_ids = series_ids[:MAX_SERIES_PER_STUDY]
        vols = [self._load_nifti(s) for s in series_ids.seriesId]
        vol = torch.stack(vols, dim=0)

        label = torch.tensor(label, dtype=torch.long)

        return vol, label


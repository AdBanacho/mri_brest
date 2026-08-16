import os
import hashlib
import re
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

from mriBreastDuke.constants import (
    NIFTI_PATH,
    PREPARED_TO_TRAIN_PATH,
    SIZE_CACHE_PATH,
    SUBTRACTION_PATH,
)
from mriBreastDuke.dataLoaders.subtraction import (
    SUBTRACTION_NONE,
    build_subtraction_pairs,
    validate_subtraction_mode,
)


class NiftiDataset(Dataset):
    def __init__(self, df,
                 target_size=None,
                 image_root=NIFTI_PATH,
                 target_col="label",
                 size_cache_path='train',
                 use_monai=True,
                 subtraction_mode=SUBTRACTION_NONE,
                 subtraction_root=SUBTRACTION_PATH):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_col = target_col
        self.target_size = target_size
        self.size_cache_path = SIZE_CACHE_PATH(size_cache_path)
        # self.sizes = load_or_compute_sizes(self.df, self.size_cache_path, self.image_root, self.serie_col)
        self.use_monai = use_monai
        self.subtraction_mode = validate_subtraction_mode(subtraction_mode)
        self.subtraction_root = subtraction_root

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

    def _target_size_cache_key(self):
        if self.target_size is None:
            return "original_size"
        return "size_" + "x".join(str(value) for value in self.target_size)

    @staticmethod
    def _safe_path_component(value, max_length=48):
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
        return (value or "unknown")[:max_length]

    def _subtraction_cache_path(
        self,
        study_id,
        pair_index,
        minuend_series_id,
        subtrahend_series_id,
    ):
        pair_key = f"{minuend_series_id}|{subtrahend_series_id}"
        digest = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:16]
        filename = (
            f"{pair_index:02d}_"
            f"{self._safe_path_component(minuend_series_id)}_minus_"
            f"{self._safe_path_component(subtrahend_series_id)}_"
            f"{digest}.nii.gz"
        )
        return os.path.join(
            self.subtraction_root,
            self.subtraction_mode,
            self._target_size_cache_key(),
            self._safe_path_component(study_id),
            filename,
        )

    def _load_source_for_subtraction(self, series_id):
        path = os.path.join(self.image_root, f"{series_id}.nii.gz")
        source = nib.load(path)
        source_shape = tuple(np.squeeze(source.shape))
        source_affine = np.asarray(source.affine)

        transforms = [
            LoadImaged(keys="img", image_only=False),
            EnsureChannelFirstd(keys="img"),
        ]
        if self.target_size is not None:
            transforms.append(
                ResizeD(
                    keys="img",
                    spatial_size=self.target_size,
                    mode="trilinear",
                    align_corners=False,
                )
            )

        output = Compose(transforms)({"img": path})
        transformed_affine = np.asarray(
            output["img_meta_dict"].get("affine", source_affine)
        )
        return (
            output["img"].float(),
            source_shape,
            source_affine,
            transformed_affine,
        )

    @staticmethod
    def _validate_subtraction_geometry(
        minuend_shape,
        minuend_affine,
        subtrahend_shape,
        subtrahend_affine,
        minuend_series_id,
        subtrahend_series_id,
    ):
        same_shape = minuend_shape == subtrahend_shape
        same_affine = np.allclose(minuend_affine, subtrahend_affine, rtol=1e-4, atol=1e-3)
        if not same_shape or not same_affine:
            raise ValueError(
                "Cannot subtract MRI series with different geometry: "
                f"{minuend_series_id} has shape {minuend_shape}, while "
                f"{subtrahend_series_id} has shape {subtrahend_shape}; "
                f"affines_match={same_affine}. "
                "Register the volumes to the same physical space before subtraction."
            )

    def _load_or_create_subtraction(
        self,
        study_id,
        pair_index,
        minuend_series_id,
        subtrahend_series_id,
    ):
        output_path = self._subtraction_cache_path(
            study_id,
            pair_index,
            minuend_series_id,
            subtrahend_series_id,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lock = FileLock(output_path + ".lock")

        if os.path.exists(output_path):
            return self._monai_load_cached(output_path)

        with lock:
            if os.path.exists(output_path):
                return self._monai_load_cached(output_path)

            (
                minuend,
                minuend_shape,
                minuend_affine,
                transformed_affine,
            ) = self._load_source_for_subtraction(minuend_series_id)
            (
                subtrahend,
                subtrahend_shape,
                subtrahend_affine,
                _,
            ) = self._load_source_for_subtraction(subtrahend_series_id)
            self._validate_subtraction_geometry(
                minuend_shape,
                minuend_affine,
                subtrahend_shape,
                subtrahend_affine,
                minuend_series_id,
                subtrahend_series_id,
            )

            difference = minuend - subtrahend
            difference = (difference - difference.mean()) / (
                difference.std(unbiased=False) + 1e-5
            )
            nib.save(
                nib.Nifti1Image(
                    difference.squeeze(0).cpu().numpy(),
                    transformed_affine,
                ),
                output_path,
            )
            return difference

    def _load_subtractions(self, row, series_ids):
        pairs = build_subtraction_pairs(series_ids, self.subtraction_mode)
        study_id = row.get("studyId", "unknown_study")
        return [
            self._load_or_create_subtraction(
                study_id,
                pair_index,
                minuend_series_id,
                subtrahend_series_id,
            )
            for pair_index, (minuend_series_id, subtrahend_series_id) in enumerate(pairs)
        ]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        series_ids = row["series_ids"]
        label = torch.tensor(row[self.target_col], dtype=torch.long)

        if self.subtraction_mode == SUBTRACTION_NONE:
            vols = [self._load_nifti(sid) for sid in series_ids]
        else:
            vols = self._load_subtractions(row, series_ids)

        vol = torch.stack(vols, dim=0)

        return vol, label

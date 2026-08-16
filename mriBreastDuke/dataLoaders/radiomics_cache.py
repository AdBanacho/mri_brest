"""Create and reuse a process-safe CSV cache of MRI-derived features."""

import os
from pathlib import Path
import time

import pandas as pd

from mriBreastDuke.dataLoaders.features import extract_feature_table


def _cache_exists(path):
    return path.is_file() and path.stat().st_size > 0


class _ExclusiveFileLock:
    """Small Linux file lock without an additional runtime dependency."""

    def __init__(self, path, timeout):
        self.path = Path(path)
        self.timeout = timeout
        self._handle = None

    def __enter__(self):
        import fcntl

        self._handle = self.path.open("a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(
                    self._handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(
                        f"Timed out waiting for feature-cache lock: {self.path}"
                    )
                time.sleep(0.25)

    def __exit__(self, exc_type, exc_value, traceback):
        import fcntl

        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _resolve_relative_paths(values, base_directory):
    def resolve(value):
        if pd.isna(value):
            return value
        path = Path(os.fspath(value))
        return os.fspath(path if path.is_absolute() else base_directory / path)

    return values.map(resolve)


def _attach_lesion_mask_paths(
    studies,
    merge_key,
    mask_path_column,
    lesion_masks_csv=None,
    lesion_mask_root=None,
    lesion_mask_suffix=".nii.gz",
    mask_transform_column=None,
):
    if mask_path_column in studies.columns:
        prepared = studies.copy()
    elif lesion_masks_csv:
        mapping_path = Path(lesion_masks_csv).expanduser().resolve()
        mapping = pd.read_csv(mapping_path)
        required = {merge_key, mask_path_column}
        if mask_transform_column:
            required.add(mask_transform_column)
        missing = required.difference(mapping.columns)
        if missing:
            raise ValueError(
                f"Lesion-mask table is missing columns: {sorted(missing)}"
            )
        if mapping[merge_key].duplicated().any():
            raise ValueError(
                f"Lesion-mask table contains duplicate '{merge_key}' values."
            )

        selected_columns = [merge_key, mask_path_column]
        if mask_transform_column:
            selected_columns.append(mask_transform_column)
        mapping = mapping[selected_columns].copy()
        mapping[mask_path_column] = _resolve_relative_paths(
            mapping[mask_path_column],
            mapping_path.parent,
        )
        if mask_transform_column:
            mapping[mask_transform_column] = _resolve_relative_paths(
                mapping[mask_transform_column],
                mapping_path.parent,
            )
        prepared = studies.merge(
            mapping,
            on=merge_key,
            how="left",
            validate="one_to_one",
        )
    elif lesion_mask_root:
        if merge_key not in studies.columns:
            raise ValueError(f"Studies table does not contain '{merge_key}'.")
        mask_root = Path(lesion_mask_root).expanduser().resolve()
        prepared = studies.copy()
        prepared[mask_path_column] = prepared[merge_key].map(
            lambda identifier: os.fspath(
                mask_root / f"{identifier}{lesion_mask_suffix}"
            )
        )
    else:
        raise ValueError(
            "The radiomics cache does not exist and lesion-mask locations are "
            "unknown. Provide --lesion_masks_csv or --lesion_mask_root."
        )

    missing_mask = prepared[mask_path_column].isna()
    if missing_mask.any():
        identifiers = prepared.loc[missing_mask, merge_key].astype(str).tolist()
        preview = ", ".join(identifiers[:5])
        raise ValueError(
            f"Missing lesion-mask paths for {missing_mask.sum()} studies, "
            f"including: {preview}"
        )
    return prepared


def ensure_radiomics_cache(
    studies,
    cache_path,
    image_root,
    merge_key="studyId",
    mask_path_column="lesion_mask_path",
    lesion_masks_csv=None,
    lesion_mask_root=None,
    lesion_mask_suffix=".nii.gz",
    registered_mask_root=None,
    mask_transform_column=None,
    lock_timeout=21600,
    extractor=extract_feature_table,
):
    """Return an existing feature CSV or extract it once under a file lock.

    Every extraction computes kinetic, morphology, and heterogeneity features.
    Selection of feature families happens later in the training workflow.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    if _cache_exists(cache_path):
        print(f"[FEATURE CACHE] Reusing {cache_path}", flush=True)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{cache_path}.lock")
    lock = _ExclusiveFileLock(lock_path, timeout=lock_timeout)
    print(f"[FEATURE CACHE] Waiting for lock {lock_path}", flush=True)
    with lock:
        if _cache_exists(cache_path):
            print(f"[FEATURE CACHE] Reusing {cache_path}", flush=True)
            return cache_path

        extraction_studies = _attach_lesion_mask_paths(
            studies,
            merge_key=merge_key,
            mask_path_column=mask_path_column,
            lesion_masks_csv=lesion_masks_csv,
            lesion_mask_root=lesion_mask_root,
            lesion_mask_suffix=lesion_mask_suffix,
            mask_transform_column=mask_transform_column,
        )
        registered_root = Path(
            registered_mask_root
            or cache_path.parent / "registered_lesion_masks"
        ).expanduser().resolve()

        print(
            f"[FEATURE CACHE] Extracting {len(extraction_studies)} studies; "
            f"registered masks: {registered_root}",
            flush=True,
        )
        features = extractor(
            extraction_studies,
            image_root=os.fspath(Path(image_root).expanduser()),
            mask_path_column=mask_path_column,
            register_masks_if_needed=True,
            registered_mask_root=os.fspath(registered_root),
            mask_transform_column=mask_transform_column,
        )
        temporary_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        try:
            features.to_csv(temporary_path, index=False)
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        print(
            f"[FEATURE CACHE] Saved {len(features)} rows to {cache_path}",
            flush=True,
        )
        return cache_path

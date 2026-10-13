"""Publish complete NIfTI caches and recover interrupted writes under a lock."""

import gzip
import os
from pathlib import Path
import tempfile
import warnings
import zlib

from filelock import FileLock
import nibabel as nib


def atomic_save_nifti(image, path):
    """Write beside the destination, then atomically publish the closed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".nii.gz", dir=path.parent,
    )
    os.close(fd)
    try:
        nib.save(image, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _is_corrupt_cache(error):
    """Recognize read failures, including errors wrapped by MONAI transforms."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (EOFError, gzip.BadGzipFile, zlib.error,
                              nib.filebasedimages.ImageFileError)):
            return True
        # nibabel uses OSError for an uncompressed payload shorter than its
        # header promises. Other I/O failures (e.g. permissions) must propagate.
        if isinstance(error, OSError):
            message = str(error).lower()
            if message.startswith("expected ") and "bytes" in message and "got " in message:
                return True
        error = error.__cause__ or error.__context__
    return False


def load_or_create_nifti_cache(path, load, create, *, lock=None, source):
    """Read or rebuild one derived cache, never suppressing source failures.

    ``load`` must read the voxel data eagerly so truncated payloads are caught.
    ``create`` must publish with ``atomic_save_nifti`` and return the volume.
    All readers use the writer's lock, including when the cache already exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        lock = FileLock(str(path) + ".lock")
    with lock:
        if path.exists():
            try:
                return load(str(path))
            except Exception as error:
                if not _is_corrupt_cache(error):
                    raise RuntimeError(f"Cannot read MRI cache {path}") from error
                warnings.warn(
                    f"Rebuilding corrupt MRI cache {path} from {source}: {error}",
                    RuntimeWarning,
                )
        try:
            return create()
        except Exception as error:
            raise RuntimeError(
                f"Cannot create MRI cache {path} from source {source}"
            ) from error

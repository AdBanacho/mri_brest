"""Real NIfTI cache tests without importing the GPU training stack."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np

MODULE_PATH = (Path(__file__).resolve().parents[1] / "mriBreastDuke"
               / "dataLoaders" / "nifti_cache.py")
SPEC = importlib.util.spec_from_file_location("nifti_cache", MODULE_PATH)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


def load_volume(path):
    return nib.load(path).get_fdata()


def concurrent_reader(path, start, counter, results):
    try:
        if not start.wait(10):
            raise RuntimeError("start gate timed out")

        def create():
            with counter.get_lock():
                counter.value += 1
            time.sleep(0.1)
            image = nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4))
            CACHE.atomic_save_nifti(image, path)
            return image.get_fdata()

        volume = CACHE.load_or_create_nifti_cache(
            path, load_volume, create, source="source.nii.gz",
        )
        results.put((volume.shape, float(volume.sum())))
    except Exception as error:
        results.put(str(error))


class NiftiCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "cached.nii.gz"
        self.data = np.random.default_rng(42).normal(size=(16, 16, 8)).astype(np.float32)
        self.image = nib.Nifti1Image(self.data, np.eye(4))

    def create(self):
        CACHE.atomic_save_nifti(self.image, self.path)
        return self.data

    def truncate_cache(self):
        nib.save(self.image, self.path)
        payload = self.path.read_bytes()
        self.path.write_bytes(payload[:len(payload) // 2])

    def test_truncated_gzip_cache_is_rebuilt_from_source(self):
        self.truncate_cache()
        with self.assertRaises(EOFError):
            load_volume(self.path)
        with self.assertWarnsRegex(RuntimeWarning, "Rebuilding corrupt MRI cache"):
            result = CACHE.load_or_create_nifti_cache(
                self.path, load_volume, self.create, source="source.nii.gz",
            )
        np.testing.assert_array_equal(result, self.data)
        np.testing.assert_array_equal(load_volume(self.path), self.data)

    def test_monai_wrapped_eof_is_also_rebuilt(self):
        self.truncate_cache()

        def wrapped_load(path):
            try:
                return load_volume(path)
            except EOFError as error:
                raise RuntimeError("applying transform LoadImaged") from error

        with self.assertWarns(RuntimeWarning):
            result = CACHE.load_or_create_nifti_cache(
                self.path, wrapped_load, self.create, source="source.nii.gz",
            )
        np.testing.assert_array_equal(result, self.data)

    def test_healthy_cache_is_reused(self):
        self.create()
        with patch.object(self, "create", side_effect=AssertionError("must reuse")):
            result = CACHE.load_or_create_nifti_cache(
                self.path, load_volume, self.create, source="source.nii.gz",
            )
        np.testing.assert_array_equal(result, self.data)

    def test_existing_cache_reader_waits_for_writer_lock(self):
        self.create()
        entered = threading.Event()
        read = threading.Event()

        def load(path):
            read.set()
            return load_volume(path)

        def reader():
            entered.set()
            return CACHE.load_or_create_nifti_cache(
                self.path, load, self.create, source="source.nii.gz",
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            with CACHE.FileLock(str(self.path) + ".lock"):
                future = pool.submit(reader)
                self.assertTrue(entered.wait(5))
                self.assertFalse(read.wait(0.2))
            np.testing.assert_array_equal(future.result(timeout=5), self.data)

    def test_concurrent_processes_create_only_one_complete_cache(self):
        context = multiprocessing.get_context("fork")
        start = context.Event()
        counter = context.Value("i", 0)
        results = context.Queue()
        workers = [context.Process(target=concurrent_reader,
                                   args=(self.path, start, counter, results))
                   for _ in range(8)]
        try:
            for worker in workers:
                worker.start()
            start.set()
            for _ in workers:
                self.assertEqual(results.get(timeout=15), ((8, 8, 8), 512.0))
            for worker in workers:
                worker.join(timeout=5)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(counter.value, 1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
            results.close()

    def test_partial_write_is_never_published(self):
        started = threading.Event()
        finish = threading.Event()
        payload = gzip.compress(self.image.to_bytes())

        def staged_save(image, temporary):
            with open(temporary, "wb") as stream:
                stream.write(payload[:len(payload) // 2])
                stream.flush()
                started.set()
                if not finish.wait(5):
                    raise RuntimeError("test timed out")
                stream.write(payload[len(payload) // 2:])

        with patch.object(CACHE.nib, "save", side_effect=staged_save):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(CACHE.atomic_save_nifti, self.image, self.path)
                try:
                    self.assertTrue(started.wait(5))
                    self.assertFalse(self.path.exists())
                finally:
                    finish.set()
                future.result(timeout=5)
        np.testing.assert_array_equal(load_volume(self.path), self.data)

    def test_failed_write_preserves_previous_cache_and_cleans_temporary(self):
        self.create()
        original = self.path.read_bytes()

        def interrupted_save(image, temporary):
            Path(temporary).write_bytes(b"unfinished")
            raise OSError("disk full")

        with patch.object(CACHE.nib, "save", side_effect=interrupted_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                CACHE.atomic_save_nifti(self.image, self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_corrupt_source_is_not_hidden_or_published(self):
        self.truncate_cache()
        original = self.path.read_bytes()
        source = self.root / "source.nii.gz"
        source.write_bytes(original)

        def create_from_bad_source():
            image = nib.Nifti1Image(load_volume(source), np.eye(4))
            CACHE.atomic_save_nifti(image, self.path)

        with self.assertWarns(RuntimeWarning):
            with self.assertRaisesRegex(RuntimeError, str(source)) as failure:
                CACHE.load_or_create_nifti_cache(
                    self.path, load_volume, create_from_bad_source, source=str(source),
                )
        self.assertIsInstance(failure.exception.__cause__, EOFError)
        self.assertEqual(self.path.read_bytes(), original)

    def test_permission_errors_do_not_trigger_regeneration(self):
        self.create()
        create = Mock()
        with self.assertRaisesRegex(RuntimeError, str(self.path)) as failure:
            CACHE.load_or_create_nifti_cache(
                self.path, Mock(side_effect=PermissionError("denied")),
                create, source="source.nii.gz",
            )
        create.assert_not_called()
        self.assertIsInstance(failure.exception.__cause__, PermissionError)


if __name__ == "__main__":
    unittest.main()

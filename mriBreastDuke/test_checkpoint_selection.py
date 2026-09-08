import tempfile
import unittest
from pathlib import Path

from mriBreastDuke.checkpoint_selection import rank_checkpoints


class BestCheckpointSelectionTests(unittest.TestCase):
    def _checkpoint_dir(self, names):
        temporary_directory = tempfile.TemporaryDirectory()
        path = Path(temporary_directory.name)
        for name in names:
            (path / name).touch()
        self.addCleanup(temporary_directory.cleanup)
        return path

    def test_auc_checkpoint_prefers_auc_over_balanced_accuracy(self):
        checkpoint_dir = self._checkpoint_dir(
            [
                "best-epoch=01-val_auc_roc=0.7000-"
                "val_balanced_accuracy=0.9000-val_sensitivity=0.9500.ckpt",
                "best-epoch=02-val_auc_roc=0.8500-"
                "val_balanced_accuracy=0.7000-val_sensitivity=0.7500.ckpt",
            ]
        )

        selected = rank_checkpoints(checkpoint_dir.glob("best-*.ckpt"))[0]

        self.assertIn("val_auc_roc=0.8500", selected.name)

    def test_prefers_auc_checkpoint_over_older_formats(self):
        checkpoint_dir = self._checkpoint_dir(
            [
                "best-epoch=01-val_sensitivity=1.0000-val_auc_roc=0.9900.ckpt",
                "best-epoch=02-val_balanced_accuracy=0.9500-"
                "val_sensitivity=0.9500-val_auc_roc=0.9900.ckpt",
                "best-epoch=03-val_auc_roc=0.7000-"
                "val_balanced_accuracy=0.6500-val_sensitivity=0.7000.ckpt",
            ]
        )

        selected = rank_checkpoints(checkpoint_dir.glob("best-*.ckpt"))[0]

        self.assertIn("best-epoch=03-val_auc_roc=0.7000", selected.name)

    def test_balanced_checkpoints_still_prefer_balanced_accuracy(self):
        checkpoint_dir = self._checkpoint_dir(
            [
                "best-epoch=01-val_balanced_accuracy=0.6000-"
                "val_sensitivity=1.0000-val_auc_roc=0.5500.ckpt",
                "best-epoch=02-val_balanced_accuracy=0.7500-"
                "val_sensitivity=0.8000-val_auc_roc=0.8000.ckpt",
            ]
        )

        selected = rank_checkpoints(checkpoint_dir.glob("best-*.ckpt"))[0]

        self.assertIn("val_balanced_accuracy=0.7500", selected.name)

    def test_prefers_new_balanced_checkpoint_over_legacy_checkpoint(self):
        checkpoint_dir = self._checkpoint_dir(
            [
                "best-epoch=01-val_sensitivity=1.0000-val_auc_roc=0.9000.ckpt",
                "best-epoch=02-val_balanced_accuracy=0.6500-"
                "val_sensitivity=0.7500-val_auc_roc=0.7000.ckpt",
            ]
        )

        selected = rank_checkpoints(checkpoint_dir.glob("best-*.ckpt"))[0]

        self.assertIn("val_balanced_accuracy=0.6500", selected.name)

    def test_legacy_checkpoints_still_use_sensitivity_then_auc(self):
        checkpoint_dir = self._checkpoint_dir(
            [
                "best-epoch=01-val_sensitivity=0.9000-val_auc_roc=0.8000.ckpt",
                "best-epoch=02-val_sensitivity=0.9000-val_auc_roc=0.8500-v1.ckpt",
                "best-epoch=03-val_sensitivity=0.8500-val_auc_roc=0.9500.ckpt",
            ]
        )

        selected = rank_checkpoints(checkpoint_dir.glob("best-*.ckpt"))[0]

        self.assertEqual(
            selected.name,
            "best-epoch=02-val_sensitivity=0.9000-val_auc_roc=0.8500-v1.ckpt",
        )


if __name__ == "__main__":
    unittest.main()

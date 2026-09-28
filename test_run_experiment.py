"""Checks for array planning and handoff between training and validation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment as runner


class ExperimentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = runner.load_config(Path(__file__).with_name("experiment.toml"))

    def test_training_and_validation_share_ids(self):
        self.assertEqual(len(runner._train_configs(self.config)), 48)
        self.assertEqual(len(runner._validation_configs(self.config)), 48)
        self.assertIn("resnet10", {c["mri_model"] for c in runner._train_configs(self.config)})
        for index in range(48):
            train = runner.task_command(self.config, "train", index)
            validate = runner.task_command(self.config, "validate", index)
            self.assertEqual(train[train.index("--experiment_id") + 1],
                             validate[validate.index("--experiment_id") + 1])

    def test_cap_and_validation_alpha(self):
        config = copy.deepcopy(self.config)
        config["jobs"]["max_jobs"] = 2
        config["validate"]["grid"]["fusion_alpha"] = [0.3, 0.5]
        self.assertEqual(len(runner._train_configs(config)), 2)
        self.assertEqual(len(runner._validation_configs(config)), 4)
        self.assertEqual(runner.task_command(config, "validate", 1)[
            runner.task_command(config, "validate", 1).index("--fusion_alpha") + 1], "0.5")

    def test_other_hyperparameters_change_checkpoint_identity(self):
        a = runner._train_configs(self.config)[0]
        b = {**a, "xgb_extra": {"reg_lambda": 2.0}}
        self.assertNotEqual(runner._experiment_id(self.config, a),
                            runner._experiment_id(self.config, b))

    def test_summary_uses_hyphenated_flags(self):
        command = runner.task_command(self.config, "summarize", 0)
        self.assertIn("--input-dir", command)
        self.assertIn("--expected-folds", command)
        self.assertNotIn("--input_dir", command)

    def test_submitted_array_uses_dynamic_count_and_concurrency(self):
        config = copy.deepcopy(self.config)
        config["jobs"]["max_jobs"] = 3
        config["jobs"]["max_concurrent"] = 2
        config["cleanup"] = {"before_train": False, "before_validate": False}
        with patch.object(runner.subprocess, "run") as run:
            run.return_value.stdout = "12345\n"
            runner.submit(config, Path("experiment.toml").resolve(), "train", False)
        args, kwargs = run.call_args
        self.assertIn("--array=0-2%2", args[0])
        self.assertIn('"${SLURM_ARRAY_TASK_ID:-0}"', kwargs["input"])
        self.assertIn("--snapshot", kwargs["input"])

    def test_cleanup_runs_once_before_train_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(self.config)
            config["jobs"]["max_jobs"] = 1
            for name, value in (("checkpoint_root", "checkpoints"),
                                ("logs_root", "lightning"),
                                ("validation_root", "validation"),
                                ("slurm_logs", "slurm")):
                config["paths"][name] = str(root / value)
                (root / value).mkdir()
                (root / value / "keep.txt").write_text("sample")

            def submitted(*args, **kwargs):
                self.assertFalse((root / "checkpoints").exists())
                self.assertFalse((root / "lightning").exists())
                self.assertTrue((root / "validation" / "keep.txt").exists())
                self.assertEqual(len(list((root / "slurm").glob("experiment-*.json"))), 1)
                return type("Result", (), {"stdout": "12345\n"})()

            with patch.object(runner.subprocess, "run", side_effect=submitted) as run:
                runner.submit(config, Path("experiment.toml"), "train", False)
            run.assert_called_once()

    def test_validate_cleanup_removes_summary_and_dry_run_keeps_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(self.config)
            config["jobs"]["max_jobs"] = 1
            config["paths"].update({
                "checkpoint_root": str(root / "checkpoints"),
                "logs_root": str(root / "lightning"),
                "validation_root": str(root / "validation"),
                "summary_root": str(root / "other_summary"),
                "slurm_logs": str(root / "slurm"),
            })
            for name in ("checkpoints", "lightning", "validation", "other_summary"):
                (root / name).mkdir()
                (root / name / "keep.txt").write_text("sample")
            runner.submit(config, Path("experiment.toml"), "validate", True)
            self.assertTrue((root / "validation" / "keep.txt").exists())
            with patch.object(runner.subprocess, "run") as run:
                run.return_value.stdout = "12345\n"
                runner.submit(config, Path("experiment.toml"), "validate", False)
            self.assertFalse((root / "validation").exists())
            self.assertFalse((root / "other_summary").exists())
            self.assertTrue((root / "checkpoints" / "keep.txt").exists())

    def test_cleanup_rejects_input_directory(self):
        config = copy.deepcopy(self.config)
        config["paths"]["logs_root"] = config["paths"]["image_root"]
        with self.assertRaisesRegex(ValueError, "configured input"):
            runner._cleanup_targets(config, "train")


if __name__ == "__main__":
    unittest.main()

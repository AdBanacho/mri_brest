"""Checks for array planning and handoff between training and validation."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment as runner


class ExperimentRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = runner.load_config(Path(__file__).with_name("experiment.toml"))

    def test_training_and_validation_share_ids(self):
        self.assertEqual(len(runner._train_configs(self.config)), 32)
        self.assertEqual(len(runner._validation_configs(self.config)), 32)
        for index in range(32):
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
        with patch.object(runner.subprocess, "run") as run:
            run.return_value.stdout = "12345\n"
            runner.submit(config, Path("experiment.toml").resolve(), "train", False)
        args, kwargs = run.call_args
        self.assertIn("--array=0-2%2", args[0])
        self.assertIn('"${SLURM_ARRAY_TASK_ID:-0}"', kwargs["input"])
        self.assertIn("--snapshot", kwargs["input"])


if __name__ == "__main__":
    unittest.main()

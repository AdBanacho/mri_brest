import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .xgboost_fusion import probability_metrics, save_fusion_predictions


class FusionOutputTest(unittest.TestCase):
    def test_uses_tabular_model_name_in_prediction_columns(self):
        probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
        validation_data = pd.DataFrame(
            {"patientId": ["p1", "p2"], "studyId": ["s1", "s2"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.csv"
            save_fusion_predictions(
                validation_data,
                np.array([0, 1]),
                probabilities,
                probabilities,
                probabilities,
                output_path,
                tabular_model_name="mlp",
            )
            output = pd.read_csv(output_path)

        self.assertIn("mlp_probability_1", output.columns)
        self.assertIn("mlp_prediction", output.columns)
        self.assertNotIn("xgboost_prediction", output.columns)

    def test_saves_predictions_from_branch_specific_thresholds(self):
        probabilities = np.array([[0.4, 0.6], [0.2, 0.8]])
        validation_data = pd.DataFrame(
            {"patientId": ["p1", "p2"], "studyId": ["s1", "s2"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.csv"
            save_fusion_predictions(
                validation_data,
                np.array([0, 1]),
                probabilities,
                probabilities,
                probabilities,
                output_path,
                decision_thresholds={
                    "image": 0.8,
                    "xgboost": 0.8,
                    "fusion": 0.8,
                },
            )
            output = pd.read_csv(output_path)

        self.assertEqual(output["image_prediction"].tolist(), [0, 1])
        self.assertEqual(output["fusion_prediction"].tolist(), [0, 1])
        self.assertTrue((output["fusion_decision_threshold"] == 0.8).all())

    def test_binary_sensitivity_is_positive_class_recall(self):
        probabilities = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]
        )
        metrics = probability_metrics(
            np.array([0, 0, 1, 1]), probabilities, prefix="image"
        )

        self.assertEqual(metrics["image_sensitivity"], 0.5)
        self.assertEqual(metrics["image_specificity"], 1.0)

    def test_binary_metrics_apply_supplied_decision_threshold(self):
        probabilities = np.array(
            [[0.4, 0.6], [0.35, 0.65], [0.2, 0.8], [0.1, 0.9]]
        )

        metrics = probability_metrics(
            np.array([0, 0, 1, 1]),
            probabilities,
            prefix="fusion",
            threshold=0.8,
        )

        self.assertEqual(metrics["fusion_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["fusion_decision_threshold"], 0.8)

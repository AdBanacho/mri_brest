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

    def test_binary_sensitivity_is_positive_class_recall(self):
        probabilities = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.9]]
        )
        metrics = probability_metrics(
            np.array([0, 0, 1, 1]), probabilities, prefix="image"
        )

        self.assertEqual(metrics["image_sensitivity"], 0.5)

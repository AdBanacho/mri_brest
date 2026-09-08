import unittest

import numpy as np

from mriBreastDuke.threshold_tuning import (
    make_inner_calibration_split,
    predictions_at_threshold,
    select_balanced_accuracy_threshold,
)


class InnerCalibrationSplitTests(unittest.TestCase):
    def test_patient_groups_never_cross_the_inner_split(self):
        groups = np.repeat([f"patient-{index}" for index in range(20)], 2)
        labels = np.repeat(np.tile([0, 1], 10), 2)

        fit_indices, calibration_indices, effective_folds = (
            make_inner_calibration_split(
                labels,
                groups=groups,
                requested_folds=5,
                random_state=43,
            )
        )

        self.assertEqual(effective_folds, 5)
        self.assertFalse(
            set(groups[fit_indices]).intersection(groups[calibration_indices])
        )
        self.assertEqual(set(labels[fit_indices]), {0, 1})
        self.assertEqual(set(labels[calibration_indices]), {0, 1})

    def test_split_is_deterministic(self):
        groups = np.array([f"patient-{index}" for index in range(20)])
        labels = np.tile([0, 1], 10)

        first = make_inner_calibration_split(labels, groups, 5, 47)
        second = make_inner_calibration_split(labels, groups, 5, 47)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class DecisionThresholdTests(unittest.TestCase):
    def test_tunes_threshold_away_from_point_five(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.60, 0.65, 0.80, 0.90])

        result = select_balanced_accuracy_threshold(labels, probabilities)

        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        np.testing.assert_array_equal(
            predictions_at_threshold(probabilities, result["threshold"]),
            labels,
        )

    def test_rejects_non_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "binary labels"):
            select_balanced_accuracy_threshold(
                np.array([0, 1, 2]),
                np.array([0.1, 0.5, 0.9]),
            )


if __name__ == "__main__":
    unittest.main()

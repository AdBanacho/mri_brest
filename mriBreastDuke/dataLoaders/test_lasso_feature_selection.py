import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("lasso_feature_selection.py")
SPEC = importlib.util.spec_from_file_location("lasso_feature_selection", MODULE_PATH)
LASSO_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LASSO_MODULE)
LassoFeatureSelector = LASSO_MODULE.LassoFeatureSelector


class LassoFeatureSelectorTest(unittest.TestCase):
    def test_selects_signal_and_transforms_with_patient_grouped_cv(self):
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(40), 2)
        labels = np.repeat(np.tile([0, 1], 20), 2)
        signal = (2 * labels - 1) + rng.normal(scale=0.15, size=len(labels))
        features = np.column_stack(
            [signal, rng.normal(size=(len(labels), 5))]
        )

        selector = LassoFeatureSelector(
            cv_folds=4,
            cs=8,
            max_iter=3000,
            random_state=7,
        ).fit(features, labels, groups=groups)

        self.assertTrue(selector.get_support()[0])
        self.assertEqual(selector.effective_cv_folds_, 4)
        self.assertEqual(
            selector.transform(features).shape,
            (len(features), selector.output_dimension),
        )

    def test_caps_inner_folds_at_available_groups_per_class(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(6), 2)
        labels = np.repeat([0, 1, 0, 1, 0, 1], 2)
        features = rng.normal(size=(12, 3))
        features[:, 0] += labels * 2

        selector = LassoFeatureSelector(
            cv_folds=5,
            cs=4,
            max_iter=2000,
            tolerance=1e-3,
        ).fit(features, labels, groups=groups)

        self.assertEqual(selector.effective_cv_folds_, 3)

    def test_report_identifies_feature_families_and_coefficients(self):
        labels = np.tile([0, 1], 20)
        features = np.column_stack(
            [
                labels * 3.0,
                np.linspace(-1, 1, len(labels)),
                np.tile([-0.2, 0.2], 20),
                np.zeros(len(labels)),
            ]
        )
        names = [
            "continuous__age_at_diagnosis_years",
            "continuous__kinetic_peak_ser",
            "continuous__morphology_BreastVol",
            "continuous__heterogeneity_entropy",
        ]

        selector = LassoFeatureSelector(
            cv_folds=4,
            cs=6,
            max_iter=2000,
        ).fit(features, labels)
        report = selector.selection_report(names)

        self.assertEqual(
            set(report["feature_group"]),
            {"clinical", "kinetic", "morphology", "heterogeneity"},
        )
        self.assertIn("coefficient_class_1", report.columns)
        self.assertTrue(report.iloc[0]["selected"])

    def test_rejects_transform_with_different_feature_count(self):
        labels = np.tile([0, 1], 10)
        features = np.column_stack([labels, 1 - labels]).astype(float)
        selector = LassoFeatureSelector(cv_folds=2, cs=3).fit(features, labels)

        with self.assertRaisesRegex(ValueError, "same columns"):
            selector.transform(np.ones((4, 3)))


if __name__ == "__main__":
    unittest.main()

"""Annotation audits without importing the training stack.

Run with: python -m unittest discover -s tests -p 'test_annotation_boxes.py'
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd


MODULE_PATH = (Path(__file__).resolve().parents[1] / "mriBreastDuke"
               / "dataLoaders" / "annotation_boxes.py")
SPEC = importlib.util.spec_from_file_location("annotation_boxes", MODULE_PATH)
ANNOTATIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANNOTATIONS)


class AnnotationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def save_image(self, series_id, shape, affine=None):
        nib.save(
            nib.Nifti1Image(np.zeros(shape, dtype=np.uint8),
                           np.eye(4) if affine is None else affine),
            self.root / f"{series_id}.nii.gz",
        )

    def coverage(self, rows, boxes):
        studies = pd.DataFrame(rows, columns=["patientId", "studyId", "series_ids"])
        return ANNOTATIONS.annotation_coverage(studies, boxes, image_root=self.root)

    def test_reported_patient_is_excluded_and_other_studies_remain_eligible(self):
        series_id = "1.3.6.1.4.1.14519.5.2.1.141926131899404629167167471770553204323"
        self.save_image(series_id, (256, 256, 96))
        self.save_image("valid", (16, 16, 8))
        boxes = ANNOTATIONS.load_annotation_boxes()
        self.assertEqual(boxes["Breast_MRI_207"], ((358, 374, 34), (365, 387, 40)))
        valid_box = ((1, 2, 3), (4, 5, 6))
        boxes["valid_patient"] = valid_box

        matched, report = self.coverage([
            ("Breast_MRI_207", "invalid_study", [series_id]),
            ("valid_patient", "valid_study", ["valid"]),
        ], boxes)

        self.assertEqual(matched, {"valid_study": (valid_box, (16, 16, 8))})
        invalid = report[report.patientId == "Breast_MRI_207"].iloc[0]
        self.assertEqual(invalid.status, "out_of_volume_box")
        self.assertIn(series_id, invalid.detail)
        self.assertIn("(256, 256, 96)", invalid.detail)
        self.assertIn(str(boxes["Breast_MRI_207"]), invalid.detail)
        report.to_csv(self.root / "annotation_coverage.csv", index=False)
        saved = pd.read_csv(self.root / "annotation_coverage.csv").iloc[0]
        self.assertEqual(saved.detail, invalid.detail)

    def test_overflow_in_any_axis_excludes_the_entire_study(self):
        self.save_image("first", (16, 16, 16))
        self.save_image("second", (8, 8, 8))
        for axis in range(3):
            with self.subTest(axis=axis):
                ends = [8, 8, 8]
                ends[axis] = 9
                matched, report = self.coverage(
                    [("patient", "study", ["first", "second"])],
                    {"patient": ((0, 0, 0), tuple(ends))},
                )
                self.assertEqual(matched, {})
                self.assertEqual(report.iloc[0].status, "out_of_volume_box")
                self.assertIn("second", report.iloc[0].detail)

    def test_exclusive_end_at_volume_edge_is_eligible(self):
        self.save_image("series", (16, 16, 8))
        box = ((0, 0, 0), (16, 16, 8))
        matched, report = self.coverage(
            [("patient", "study", ["series"])], {"patient": box},
        )
        self.assertEqual(matched["study"], (box, (16, 16, 8)))
        self.assertEqual(report.iloc[0].status, "eligible")
        self.assertEqual(report.iloc[0].detail, "")

    def test_incompatible_affines_remain_ineligible(self):
        self.save_image("first", (16, 16, 8))
        affine = np.eye(4)
        affine[0, 3] = 10
        self.save_image("second", (16, 16, 8), affine)
        matched, report = self.coverage(
            [("patient", "study", ["first", "second"])],
            {"patient": ((1, 1, 1), (2, 2, 2))},
        )
        self.assertEqual(matched, {})
        self.assertEqual(report.iloc[0].status, "incompatible_series_geometry")

    def test_no_eligible_boxes_returns_a_report(self):
        self.save_image("series", (8, 8, 8))
        matched, report = self.coverage(
            [("patient", "study", ["series"])],
            {"patient": ((9, 9, 1), (10, 10, 2))},
        )
        self.assertEqual(matched, {})
        self.assertEqual(report.status.tolist(), ["out_of_volume_box"])

    def test_malformed_workbook_coordinates_still_raise(self):
        for coordinates in ((-1, 2, 0, 2, 0, 2), (3, 2, 0, 2, 0, 2)):
            with self.subTest(coordinates=coordinates):
                workbook = pd.DataFrame(
                    [("patient", *coordinates)], columns=ANNOTATIONS.COLUMNS,
                )
                with patch.object(ANNOTATIONS.pd, "read_excel", return_value=workbook):
                    with self.assertRaises(ValueError):
                        ANNOTATIONS.load_annotation_boxes("unused.xlsx")


if __name__ == "__main__":
    unittest.main()

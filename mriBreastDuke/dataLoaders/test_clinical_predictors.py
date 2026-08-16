import importlib.util
import os
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "mriBreastDuke"
    / "dataLoaders"
    / "read_from_xlmx.py"
)
SPEC = importlib.util.spec_from_file_location("read_from_xlmx", MODULE_PATH)
CLINICAL_LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLINICAL_LOADER)

TEST_WORKBOOK = os.environ.get("DUKE_CLINICAL_FEATURES_TEST_FILE")


@unittest.skipUnless(TEST_WORKBOOK, "set DUKE_CLINICAL_FEATURES_TEST_FILE")
class ClinicalPredictorLoaderTest(unittest.TestCase):
    def test_loads_every_patient_without_dropping_the_first_row(self):
        data = CLINICAL_LOADER.read_useful_clinical_predictors(TEST_WORKBOOK)

        self.assertEqual(len(data), 922)
        self.assertEqual(data.iloc[0]["patientId"], "Breast_MRI_001")
        self.assertIn("age_at_diagnosis_years", data.columns)
        self.assertNotIn("race_ethnicity", data.columns)
        self.assertNotIn("oncotype_score", data.columns)
        self.assertNotIn("therapeutic_or_prophylactic_oophorectomy", data.columns)
        self.assertNotIn("recurrence_event", data.columns)

    def test_filters_oncotype_patients_and_preserves_existing_cutoffs(self):
        data = CLINICAL_LOADER.read_useful_clinical_predictors(
            TEST_WORKBOOK,
            oncotype_only=True,
        )
        labels = CLINICAL_LOADER.read_patient_id_for_oncotype_score_not_na(
            False,
            TEST_WORKBOOK,
        )

        self.assertEqual(len(data), 261)
        self.assertEqual(labels["oncotypeCategory"].value_counts().to_dict(), {0: 160, 1: 79, 2: 22})

    def test_sensitive_predictor_requires_explicit_opt_in(self):
        data = CLINICAL_LOADER.read_useful_clinical_predictors(
            TEST_WORKBOOK,
            include_sensitive=True,
        )

        self.assertIn("race_ethnicity", data.columns)


if __name__ == "__main__":
    unittest.main()

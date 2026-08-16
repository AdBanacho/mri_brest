import importlib.util
import tempfile
from pathlib import Path
import unittest

import numpy as np

from mriBreastDuke.dataLoaders.features import (
    extract_heterogeneity_features,
    extract_morphological_features,
    extract_mri_features,
    extract_mri_features_from_nifti,
    extract_temporal_kinetic_features,
)


class TemporalKineticFeatureTest(unittest.TestCase):
    def test_extracts_washout_curve_from_known_signal(self):
        volumes = np.stack(
            [
                np.full((3, 3, 3), 100.0),
                np.full((3, 3, 3), 150.0),
                np.full((3, 3, 3), 125.0),
            ]
        )
        mask = np.ones((3, 3, 3), dtype=bool)

        features = extract_temporal_kinetic_features(
            volumes,
            mask,
            time_points=[0.0, 2.0, 5.0],
        )

        self.assertAlmostEqual(features["early_enhancement_percent"], 50.0)
        self.assertAlmostEqual(features["late_enhancement_percent"], 25.0)
        self.assertAlmostEqual(features["maximum_wash_in_slope"], 25.0)
        self.assertAlmostEqual(features["maximum_washout_slope"], -25.0 / 3.0)
        self.assertAlmostEqual(features["signal_enhancement_ratio"], 2.0)
        self.assertAlmostEqual(features["enhancement_auc"], 162.5)
        self.assertEqual(features["kinetic_curve_type"], "washout")
        self.assertEqual(features["kinetic_curve_type_code"], 2)

    def test_rejects_non_increasing_acquisition_times(self):
        volumes = np.ones((3, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_temporal_kinetic_features(
                volumes,
                np.ones((2, 2, 2)),
                time_points=[0.0, 1.0, 1.0],
            )


class MorphologicalFeatureTest(unittest.TestCase):
    def test_uses_physical_voxel_spacing(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1:3, 1:3, 1:3] = True

        features = extract_morphological_features(mask, voxel_spacing=(1.0, 2.0, 3.0))

        self.assertEqual(features["voxel_count"], 8)
        self.assertAlmostEqual(features["volume_mm3"], 48.0)
        self.assertAlmostEqual(features["surface_area_mm2"], 88.0)
        self.assertAlmostEqual(features["bounding_box_x_mm"], 2.0)
        self.assertAlmostEqual(features["bounding_box_y_mm"], 4.0)
        self.assertAlmostEqual(features["bounding_box_z_mm"], 6.0)
        self.assertEqual(features["connected_component_count"], 1)
        self.assertAlmostEqual(features["largest_component_fraction"], 1.0)

    def test_rejects_an_empty_lesion_mask(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_morphological_features(np.zeros((3, 3, 3)))


class HeterogeneityFeatureTest(unittest.TestCase):
    def test_uniform_roi_has_no_intensity_or_glcm_variation(self):
        image = np.full((3, 3, 3), 7.0)
        mask = np.ones_like(image, dtype=bool)

        features = extract_heterogeneity_features(image, mask, levels=8)

        self.assertAlmostEqual(features["std"], 0.0)
        self.assertAlmostEqual(features["entropy"], 0.0)
        self.assertAlmostEqual(features["uniformity"], 1.0)
        self.assertAlmostEqual(features["glcm_contrast"], 0.0)
        self.assertAlmostEqual(features["glcm_homogeneity"], 1.0)

    def test_gradient_roi_has_positive_texture_contrast(self):
        image = np.indices((4, 4, 4)).sum(axis=0).astype(float)
        mask = np.ones_like(image, dtype=bool)

        features = extract_heterogeneity_features(
            image,
            mask,
            levels=8,
            intensity_range=(0.0, 9.0),
        )

        self.assertGreater(features["std"], 0.0)
        self.assertGreater(features["entropy"], 0.0)
        self.assertGreater(features["glcm_contrast"], 0.0)


class CombinedFeatureTest(unittest.TestCase):
    def test_combines_all_three_feature_families(self):
        pre = np.full((3, 3, 3), 10.0)
        post = pre.copy()
        post[1:, 1:, 1:] += 5.0
        volumes = np.stack([pre, post])
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1:, 1:, 1:] = True

        features = extract_mri_features(volumes, mask, levels=8)

        self.assertIn("kinetic_early_enhancement_percent", features)
        self.assertIn("morphology_volume_mm3", features)
        self.assertIn("heterogeneity_glcm_contrast", features)
        self.assertEqual(features["heterogeneity_source"], "first_post_subtraction")

    @unittest.skipUnless(importlib.util.find_spec("nibabel"), "nibabel is not installed")
    def test_reads_registered_nifti_series_and_header_spacing(self):
        import nibabel as nib

        affine = np.diag([1.0, 2.0, 3.0, 1.0])
        pre = np.full((3, 3, 3), 10.0, dtype=np.float32)
        post = np.full((3, 3, 3), 15.0, dtype=np.float32)
        mask = np.ones((3, 3, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = [directory / "pre.nii.gz", directory / "post.nii.gz"]
            mask_path = directory / "mask.nii.gz"
            nib.save(nib.Nifti1Image(pre, affine), paths[0])
            nib.save(nib.Nifti1Image(post, affine), paths[1])
            nib.save(nib.Nifti1Image(mask, affine), mask_path)

            features = extract_mri_features_from_nifti(paths, mask_path, levels=8)

        self.assertAlmostEqual(features["morphology_volume_mm3"], 162.0)
        self.assertAlmostEqual(features["kinetic_early_enhancement_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()

# MRI-derived feature extraction

`mriBreastDuke.features` extracts three complementary feature families from an
ordered dynamic contrast-enhanced (DCE) MRI series and a binary lesion mask.

## Requirements

- The first volume must be pre-contrast; the following volumes must be ordered
  post-contrast phases.
- Every phase and the lesion mask must be registered to the same geometry.
- Use raw or consistently bias-corrected intensities. Do not independently
  z-score each phase before extracting kinetics.
- The mask must identify the tumor/lesion. A whole-breast, fibroglandular-tissue,
  or vessel mask is not a substitute for a lesion segmentation.
- Supply acquisition times in minutes when available. Otherwise, slope and
  time-to-peak values are reported per phase index.

## Feature families

| Family | Examples |
|---|---|
| Temporal kinetics | early/late/peak enhancement, time to peak, wash-in and washout slopes, signal-enhancement ratio, enhancement AUC, curve type |
| Morphology | physical volume, surface area, sphericity, equivalent diameter, bounding-box dimensions, elongation, flatness, component count |
| Heterogeneity | percentiles, coefficient of variation, skewness, kurtosis, entropy, uniformity, and merged 3D GLCM contrast/correlation/energy/homogeneity |

## Array usage

```python
from mriBreastDuke.features import extract_mri_features

features = extract_mri_features(
    volumes,                 # shape: (time, x, y, z)
    lesion_mask,             # shape: (x, y, z)
    voxel_spacing=(0.8, 0.8, 1.5),
    time_points=[0.0, 1.4, 2.8, 5.6],
    heterogeneity_source="first_post_subtraction",
    levels=32,
    intensity_range=(-200.0, 1500.0),
)
```

Valid heterogeneity sources are `pre_contrast`, `first_post`,
`first_post_subtraction`, `late_post`, and `late_post_subtraction`.

## NIfTI usage

```python
from mriBreastDuke.features import extract_mri_features_from_nifti

features = extract_mri_features_from_nifti(
    ["pre.nii.gz", "post_1.nii.gz", "post_2.nii.gz"],
    "lesion_mask.nii.gz",
    time_points=[0.0, 1.5, 4.5],
)
```

The NIfTI loader verifies shape and affine agreement and reads voxel spacing
from the pre-contrast header.

## Dataset-level table

Add a `lesion_mask_path` column to the study DataFrame returned by the existing
loader. The `series_ids` lists must already be in chronological DCE order.

```python
from mriBreastDuke.features import extract_feature_table

feature_table = extract_feature_table(
    studies_df,
    image_root="/path/to/tciaNifti",
    mask_path_column="lesion_mask_path",
)
feature_table.to_csv("duke_mri_features.csv", index=False)
```

For final modeling, estimate imputation, scaling, feature selection, and any
cohort-wide GLCM intensity range using only the training fold. Use patient-level
splits so studies from the same patient never appear in both train and test data.

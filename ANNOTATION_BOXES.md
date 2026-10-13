# Optional Duke annotation boxes

`--use_annotation_boxes` is a Boolean CLI switch (default `false`) in both
training entry points and both checkpoint-validation entry points. The SLURM
fusion scripts expose `USE_ANNOTATION_BOXES=true|false` (default `false`).
`--annotation_boxes_file` / `ANNOTATION_BOXES_FILE` overrides the default
`mriBreastDuke/features/Annotation_Boxes.xlsx`. Disabled runs never read it,
keep their previous experiment names and outputs, and need no annotation file.

The workbook has one sheet (`Sheet1`) with `Patient ID`, `Start Row`, `End Row`,
`Start Column`, `End Column`, `Start Slice`, `End Slice`. It contains patient
IDs such as `Breast_MRI_001`, but **no study or series UID, voxel origin,
orientation, or coordinate-base declaration**. We interpret all indices as
zero-based and inclusive in the original DICOM voxel grid. SimpleITK NIfTI
axes are assumed to be (column, row, slice), the same unflipped slice order
as the workbook. This convention must be visually checked against a few raw
DICOM/NIfTI overlays before results are used scientifically. Shape/affine
checks cannot prove that an unspecified source series used the same orientation
or slice numbering. If those assumptions fail, do not enable this option until
the workbook is linked to an exact series and the transform is established.

The loader joins boxes by exact, stripped `patientId`. Only a patient with
exactly one study in the loaded cohort and source series with identical 3D
shapes and affines is eligible. It checks box bounds against **every** series.
Duplicate workbook patient rows and malformed coordinates raise errors.
Out-of-volume boxes receive `out_of_volume_box`, with the offending series UID,
image shape and box bounds in the report's `detail` column. These studies use
ordinary full-volume training; boxes are never clipped or rescaled to a guessed
source grid. For example, a box at columns 358–364 and rows 374–386 cannot be
applied to a `(256, 256, 96)` source image. Missing boxes, unmatched workbook IDs,
patients with multiple studies,
and incompatible series grids receive explicit statuses in
`annotation_coverage.csv`; those studies use ordinary full-volume training.
The report is saved beside each training fold's checkpoints, or at the
validation output root. No box is guessed from a series name or applied to a
different patient. Cohort-wide unused workbook entries are reported.
Training still requires at least one eligible study when this option is enabled;
if none is eligible, disable the option or supply annotations linked to the
correct source grids.

For eligible training studies, original half-open column/row/slice bounds are
scaled through the loader's axis-aligned resize to `(256, 256, 64)` using floor
for the start and ceil for the exclusive end. Z-score normalization changes
intensities only. The same geometry rule applies to matched subtraction pairs;
padding is appended at the high end of each axis, so voxel origins do not move.
The augmentation tries eight random background cuboids and zeros the first
that does not intersect the lesion box. If none fits, it leaves the volume
untouched. This supplies lesion-preserving training supervision without
changing the classifier's inputs or architecture. Validation, checkpoint
selection, threshold calibration, and inference receive full images **without
annotations**. The `_annboxes` experiment suffix prevents loading an ordinary
checkpoint by mistake. Box coverage is a data audit, not a localization or
predictive performance metric. Compare held-out results with the disabled run
to assess whether the augmentation helps.

From the repository root, examples of matching training and validation runs:

```bash
# Default behavior (disabled):
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow --feature_groups clinical
python -m mriBreastDuke.validate_configurable_imaging_features_fusion --feature_groups clinical

# Training-only augmentation; use the same flag during validation to select
# the corresponding checkpoints and write annotation_coverage.csv:
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow --feature_groups clinical --use_annotation_boxes
python -m mriBreastDuke.validate_configurable_imaging_features_fusion --feature_groups clinical --use_annotation_boxes

# On Helios, submit matching array jobs (or run the scripts with an array ID):
USE_ANNOTATION_BOXES=false sbatch ConfigurableImagingFeaturesFusionBinary.sh
USE_ANNOTATION_BOXES=false sbatch validate_configurable_imaging_features_fusion.sh
USE_ANNOTATION_BOXES=true sbatch ConfigurableImagingFeaturesFusionBinary.sh
USE_ANNOTATION_BOXES=true sbatch validate_configurable_imaging_features_fusion.sh
```

The basic `mriBreastDuke.trainingWorkflow` and
`mriBreastDuke.validate_best_checkpoints` commands accept the same two CLI
options; pass the switch to both when validating annotated training. These
commands also distinguish `_annboxes` checkpoints from unannotated ones.

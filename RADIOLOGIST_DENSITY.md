# Optional radiologist breast density feature

`Breast_Radiologist_Density_Assessments.xlsx` contains an `Assessments` sheet
with `Subject_ID` and three BI-RADS density ratings (`Radiologist A`, `B`,
and `C`). Ratings `a`, `b`, `c`, `d` are categorical, not a continuous
measurement. The column of explanatory text and blank columns are ignored.

The tabular branch accepts `--density_mode` with these choices:

| Mode | Density supplied to the model |
| --- | --- |
| `none` (default) | No density field; no workbook access; existing experiment name preserved |
| `majority` | Category chosen by at least two of the three radiologists |
| `radiologist_a`, `radiologist_b`, `radiologist_c` | That reader's category |

`--density_file` overrides the default workbook path. Training and validation
must use the same mode and file. A non-`none` mode adds exactly one categorical
column, `radiologist_density`, to the tabular feature set, whether or not
`clinical` is among `--feature_groups`. The existing fold-local imputer,
one-hot encoder, optional LASSO selector, and XGBoost/MLP model process this
column. It is never supplied to the MRI branch. Separate `_density-<mode>`
experiment names prevent loading a checkpoint from a different mode.

Matching is an exact, whitespace-trimmed `Subject_ID` to `patientId` join.
Blank/duplicate IDs and invalid ratings fail. A missing reader rating or a
three-way tie becomes the category `missing`. Patients absent from the
workbook also receive `missing`; no patient is dropped and no other patient's
rating is assigned. The printed coverage summarizes the modeled cohort.

The workbook covers **50 patients**, but only **18 of the 261 patients with a
numeric Oncotype score** have a density assessment (12 low-score and 6
high-score using the existing binary threshold). With `majority`, 243/261
patients therefore have the `missing` category. A model may exploit *whether
an assessment was recorded* rather than its value. Treat this experiment as
exploratory, report performance for rated and unrated patients separately,
and verify assessment dates precede the intended prediction time. The
workbook does not contain assessment or Oncotype result dates. Selecting the
best density mode using outer validation results would bias the estimated
performance; use inner selection or a separate held-out test set.

From the repository root:

```bash
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow \
  --feature_groups clinical kinetic morphology heterogeneity \
  --feature_model xgboost --density_mode majority

python -m mriBreastDuke.validate_configurable_imaging_features_fusion \
  --feature_groups clinical kinetic morphology heterogeneity \
  --feature_model xgboost --density_mode majority

# Off (default): omit --density_mode, or specify --density_mode none.
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow \
  --feature_groups clinical --density_mode none
```

Both SLURM launchers now sweep `DENSITY_MODES=(none majority)` as an
additional grid dimension. Submit matching training and validation arrays
with `sbatch ConfigurableImagingFeaturesFusionBinary.sh` and
`sbatch validate_configurable_imaging_features_fusion.sh`. IDs 0–31 retain
the old no-density configurations; IDs 32–63 add majority density. To run a
smaller grid, override the array range with `sbatch --array=0-31%8 ...` for
no density or `sbatch --array=32-63%8 ...` for majority density. The scripts
accept `DENSITY_FILE=/path/to/workbook.xlsx` for an alternate workbook.

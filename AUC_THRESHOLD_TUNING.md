# ROC-AUC checkpointing and leakage-safe threshold tuning

The configurable binary fusion workflow separates model ranking from the
final class decision:

1. The outer `StratifiedGroupKFold` reserves one patient-grouped fold for the
   final evaluation.
2. The remaining outer-training patients are split again. One deterministic
   inner fold becomes the calibration set; all other inner folds form the fit
   set.
3. The MRI model, tabular preprocessor, optional LASSO selector, and XGBoost or
   MLP model are fitted without the calibration or outer-validation patients.
4. MRI checkpoints are ranked by ROC-AUC on the inner calibration set.
5. Separate image, tabular, and fusion thresholds maximize balanced accuracy
   on that same calibration set.
6. Those thresholds are frozen and applied once to the untouched outer
   validation fold.

Patient IDs cannot cross the fit, calibration, or outer-validation boundaries.
Each checkpoint directory saves `inner_calibration_split.csv`, and standalone
validation refuses to continue if it cannot reconstruct that exact split.

## Configuration

The launchers use five inner folds by default, reserving approximately one
fifth of each outer-training fold for calibration:

```bash
THRESHOLD_CALIBRATION_FOLDS=5 \
  sbatch ConfigurableImagingFeaturesFusionBinary.sh

THRESHOLD_CALIBRATION_FOLDS=5 \
  sbatch validate_configurable_imaging_features_fusion.sh
```

Training and validation must use the same value. It is included in the
experiment directory name as `auc-cal5`, preventing new artifacts from being
mixed with older sensitivity- or balanced-accuracy-selected checkpoints.

## Validation outputs

Each validation fold contains:

- `decision_thresholds.json`: machine-readable image, tabular, and fusion
  thresholds selected for that fold;
- `threshold_calibration_metrics.csv`: selected thresholds and their
  calibration sensitivity, specificity, and balanced accuracy;
- `threshold_calibration_predictions.csv`: auditable inner-calibration
  probabilities and decisions;
- `fusion_validation_predictions.csv`: untouched outer-fold probabilities,
  frozen thresholds, and final predictions;
- threshold-aware confusion matrices for the image, tabular, and fusion
  branches.

The validation experiment root also contains
`threshold_calibration_metrics.csv` with all fold thresholds. Pooled confusion
matrices concatenate the already-thresholded predictions, so every row uses
only the threshold learned inside its own outer training fold.

This change requires retraining. Existing checkpoints did not reserve an inner
calibration set and therefore cannot be used for leakage-safe threshold tuning.

# LASSO tabular feature selection

The configurable MRI fusion workflow can select a sparse subset of the
clinical, kinetic, morphology, and heterogeneity predictors before fitting
XGBoost or the MLP.

For this binary classification problem, `--feature_selector lasso` uses the
classification equivalent of LASSO: logistic regression with an L1 penalty.
The regularization strength is chosen inside each outer training fold by an
inner stratified cross-validation loop. Patient groups remain intact in both
the outer and inner folds, and neither the outer validation rows nor their
labels are used to select features.

## Run it

The Slurm training and validation launchers use LASSO by default:

```bash
sbatch ConfigurableImagingFeaturesFusionBinary.sh
sbatch validate_configurable_imaging_features_fusion.sh
```

Set `FEATURE_SELECTOR=none` to reproduce the unfiltered tabular flow:

```bash
FEATURE_SELECTOR=none sbatch ConfigurableImagingFeaturesFusionBinary.sh
```

For a direct run, pass the feature families that should compete for selection:

```bash
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow \
  --feature_groups clinical kinetic morphology heterogeneity \
  --feature_model xgboost \
  --feature_selector lasso
```

The main selector controls are:

- `--lasso_cv_folds`: requested number of inner patient-grouped folds. It is
  reduced automatically when a training fold has fewer patient groups.
- `--lasso_cs`: number of inverse-regularization strengths to evaluate.
- `--lasso_min_features`: minimum number of predictors retained if the
  cross-validated solution shrinks every coefficient to zero.
- `--lasso_max_iter`, `--lasso_tolerance`, and `--lasso_n_jobs`: solver
  controls.
- `--lasso_plot_top_n`: maximum number of selected features displayed in each
  saved importance chart (default: 30).

Use exactly the same selector arguments for standalone validation. They are
part of the experiment directory name so that LASSO and non-LASSO artifacts
cannot be mixed accidentally.

## Feature-importance outputs

Each `fold_N/checkpoints` directory contains:

- `tabular_feature_selector.joblib`: fitted selector used by validation.
- `tabular_selected_feature_names.txt`: columns passed to XGBoost or the MLP.
- `lasso_feature_selection.csv`: every encoded feature, its source family,
  coefficient, absolute coefficient importance, rank, and selection flag.
- `lasso_feature_importance.png`: a horizontal bar chart of the most important
  selected features in that fold, colored by feature family.

The experiment checkpoint directory also contains:

- `lasso_feature_selection_by_fold.csv`: all fold reports in one table.
- `lasso_feature_stability.csv`: selection count/frequency and mean importance
  across folds. The frequency denominator is every outer fold, including folds
  where a rare one-hot category was absent. Rank this file by
  `selection_frequency`, then
  `mean_lasso_importance`, to identify the most stable predictors.
- `lasso_feature_stability.png`: the cross-fold mean-importance chart annotated
  with the percentage of folds in which each feature was selected.

Categorical clinical variables are one-hot encoded before selection, so their
individual levels appear as separate rows in the reports. Imaging-feature
columns retain their `kinetic_`, `morphology_`, or `heterogeneity_` prefix.

#!/usr/bin/env bash
# Edit this Bash file, then run: bash run_experiment.sh --train --dry-run
# Values are TOML literals because the Python planner accepts a generated TOML
# snapshot. Quote strings inside the Bash single quotes; arrays use TOML syntax.
# Relative paths resolve from this file's directory. @NAME uses constants.py.
# The runner reads these Bash variables; do not execute this file directly.

# [data]
declare -A CFG_data=(
  [seed]='42'
  [target_size]='[256, 256, 64]' # MRI volume (D, H, W)
)

# [paths]
declare -A CFG_paths=(
  [clinical_features_file]='"@TARGETS_FILE_NAME"'
  [metadata_file]='"@IMAGES_METADATA"' # Duke metadata workbook (may be local only)
  [imaging_features_file]='"@IMAGING_FEATURES_FILE_NAME"'
  [annotation_boxes_file]='"@ANNOTATION_BOXES_FILE_NAME"'
  [image_root]='"@NIFTI_PATH"'
  [prepared_root]='"@PREPARED_TO_TRAIN_PATH"'
  [subtraction_root]='"@SUBTRACTION_PATH"'
  [logs_root]='"@LIGHTING_LOGS"' # TensorBoard logs
  [checkpoint_root]='"@CHECKPOINTS_PATH"'
  [validation_root]='"@VALIDATION_CHART_PATH"'
  [slurm_logs]='"logs"' # Slurm stdout and stderr
)

# [cleanup]
declare -A CFG_cleanup=(
  [before_train]='true' # Once, before submission: delete all checkpoint_root and logs_root contents
  [before_validate]='true' # Once, before submission: delete validation_root, including summary
)

# [jobs]
declare -A CFG_jobs=(
  [max_jobs]='0' # 0 = every training combination; 12 = first 12
  [max_concurrent]='4' # Slurm array throttle for training and validation
  [account]='"plgvirtudrel2026-gpu-gh200"'
  [partition]='"plgrid-gpu-gh200"'
  [modules]='["ML-bundle"]'
)

# [jobs.train]
declare -A CFG_jobs_train=(
  [job_name]='"mri_train"'
  [cpus]='16'
  [gpus]='1'
  [memory]='"48G"'
  [time]='"24:00:00"'
)

# [jobs.validate]
declare -A CFG_jobs_validate=(
  [job_name]='"mri_validate"'
  [cpus]='16'
  [gpus]='1'
  [memory]='"48G"'
  [time]='"24:00:00"'
)

# [jobs.summarize]
declare -A CFG_jobs_summarize=(
  [job_name]='"mri_summarize"'
  [cpus]='2'
  [gpus]='0'
  [memory]='"8G"'
  [time]='"02:00:00"'
  [partition]='""' # Set a CPU partition if the cluster requires one
  [account]='""' # Or supply a CPU allocation when required
)

# [train]
declare -A CFG_train=(
  [epoch]='30'
  [num_folds]='5'
  [threshold_calibration_folds]='5'
  [num_workers]='4'
  [feature_selector]='"lasso"'
  [imaging_patient_id_column]='"Patient ID"'
  [allow_missing_imaging_features]='false'
  [include_sensitive]='false'
  [use_annotation_boxes]='true'
  [lasso_cv_folds]='5'
  [lasso_cs]='20'
  [lasso_max_iter]='5000'
  [lasso_tolerance]='1e-4'
  [lasso_min_features]='1'
  [lasso_n_jobs]='8'
  [lasso_plot_top_n]='30'
  [xgb_n_estimators]='300'
  [xgb_max_depth]='3'
  [xgb_learning_rate]='0.03'
  [xgb_subsample]='0.8'
  [xgb_colsample_bytree]='0.8'
  [xgb_n_jobs]='8'
  [mlp_hidden_layers]='"128,64"'
  [mlp_alpha]='1e-4'
  [mlp_learning_rate]='1e-3'
  [mlp_max_iter]='500'
)

# [train.xgb_extra]
declare -A CFG_train_xgb_extra=(
)

# [train.mlp_extra]
declare -A CFG_train_mlp_extra=(
)

# [train.mri_extra]
declare -A CFG_train_mri_extra=(
)

# [train.trainer_extra]
declare -A CFG_train_trainer_extra=(
)

# [train.grid]
declare -A CFG_train_grid=(
  [mri_model]='["densenet121", "resnet10", "resnet18"]'
  [subtraction_mode]='["none"]'
  [feature_groups]='[["clinical"], ["clinical", "kinetic", "morphology", "heterogeneity"]]'
  [feature_model]='["xgboost", "mlp"]'
  [batch_size]='[4]'
  [positive_boost]='[1.0]'
  [sensitivity_lambda]='[0.05, 0.2]'
  [lr]='[1e-4]'
)

# [validate]
declare -A CFG_validate=(
  [num_workers]='4' # May differ from training; fold and model settings are inherited
)

# [validate.grid]
declare -A CFG_validate_grid=(
  [fusion_alpha]='[0.5]' # Extra alphas reuse each trained checkpoint
)

# [summarize]
declare -A CFG_summarize=(
  [expected_folds]='5'
  [top_n]='20'
  [include_incomplete]='false'
)

# One configuration for the MRI fusion workflow

Edit [`experiment.toml`](experiment.toml), then run from the repository root:

```bash
python3 run_experiment.py --train --dry-run
python3 run_experiment.py --train
python3 run_experiment.py --validate
python3 run_experiment.py --summarize
```

The long forms `--training` and `--validation` work too. The dry run prints the
array size, Slurm request and first Python task command. Each normal command
submits Slurm jobs and prints the job ID. Complete training before validation,
and validation before summarization. To enforce the order in Slurm, submit each
stage after the previous array has completed; check `squeue -u "$USER"` and
`sacct -j <job-id>`. The summary is a single CPU job. `--dry-run` works for all
three stages. Jobs require the project dependencies to have been installed in
the compute environment (for example `pip install -e .` and the appropriate
requirements file) before submission; jobs do not run pip concurrently.
Each submission saves a resolved `experiment-<hash>.json` under `slurm_logs`;
queued tasks read this snapshot, so later TOML edits cannot change an active
array. Keep that snapshot with the corresponding results for provenance.

## Editing the grid

Values in `[train]` are fixed, while arrays in `[train.grid]` form a Cartesian
product. To try multiple LASSO settings, add `lasso_cs = [10, 20]` to the grid;
it overrides the fixed `lasso_cs` for each configuration. `feature_groups`
needs an array of arrays: `[["clinical"], ["kinetic", "morphology"]]`.
`[validate.grid]` sweeps `fusion_alpha` and optionally validation
`num_workers`. It is combined with each selected training configuration.
`max_jobs = 0` runs every training combination; a positive value selects the
first N in deterministic (sorted key) product order for **both** training and
validation. `max_concurrent` limits the active Slurm array tasks. The summary
searches all results under `validation_root`, including older runs.

All CLI parameters of the training module are represented in `[train]`,
`[train.grid]`, `[data]`, or `[paths]`. XGBoost, sklearn MLP, MONAI network and
Lightning Trainer constructor options can be supplied in the four `*_extra`
tables and swept using arrays of inline tables in `[train.grid]`. Only pass
options the selected model supports. `trainer_extra` cannot replace callbacks,
the logger or its output directory. Architecture options also need to be
compatible with the corresponding validation model. If you modify checkpoint
selection, fold splitting, model code or preprocessing itself, verify the
saved artifacts with the updated validation pipeline.

`[paths]` accepts absolute paths, `~`, paths relative to the TOML file, and
`@CONSTANT_NAME` values from `mriBreastDuke/constants.py`. Keep the clinical
and metadata workbooks available on the compute nodes; the metadata workbook
is normally local and excluded from Git. The selected clinical workbook now
provides both the labels and the predictors. `image_root`, preprocessing and
subtraction caches, TensorBoard logs, checkpoints, validation outputs, report
directory and Slurm logs have separate locations. The initial paths use the
repository's existing constants. Change account/partition for your allocation;
choose a CPU partition for the summary if required. To use a different config:

```bash
python3 run_experiment.py --config /path/to/experiment.toml --train
```

Each training configuration gets a `cfg` suffix derived from all its training
settings, data size/seed and input paths. Validation derives the same suffix.
This prevents different sweeps from sharing a checkpoint directory. Existing
checkpoints from the old shell scripts have no suffix, so retrain with this
runner to validate them through the unified config. Changing only
`fusion_alpha` creates a new validation directory and reuses training.

The original shell scripts remain available for older experiments. The new
runner is the single entry point for experiments using `experiment.toml`.

# Slurm experiment configuration

Edit `experiment.conf`, then submit from the repository:

```bash
bash run_experiment.sh --train --dry-run
bash run_experiment.sh --train
bash run_experiment.sh --validate
bash run_experiment.sh --summarize
```

`run_experiment.sh` only invokes `run_experiment.slurm`. All configuration loading,
path resolution, grid planning, resource selection, cleanup, snapshot creation,
submission and compute-task command construction are in the `.slurm` controller.
You can invoke it directly with `bash run_experiment.slurm` and the same flags.
The controller submits itself with `sbatch`; do not submit the controller directly
with `sbatch`, because resources and array sizes are calculated before submission.

No Python installation, TOML parser, Python planner, or ML package is needed on the
login node. Bash 4.3+ and standard Linux utilities (`realpath`, `sha256sum`, `sort`,
`mktemp`) are required. The existing Python training, validation and summary
modules still run **inside compute jobs**, after the configured modules are loaded.
`pyproject.toml` remains only as Python package build metadata, not experiment
configuration. Its obsolete `tomli` dependency has been removed.

## One editable configuration

`experiment.conf` contains ordinary shell scalars, indexed arrays, and associative
arrays. It is trusted shell code: only source configurations you trust. Paths
resolve against the configuration file's directory. All dataset/cache/output paths
are explicit; the controller does not import or parse `constants.py`.

| Setting | Purpose |
| --- | --- |
| `PATHS` | Workbooks, MRI files, prepared caches, checkpoints and output folders |
| `JOB_TRAIN`, `JOB_VALIDATE`, `JOB_SUMMARIZE` | Account, partition, CPU/GPU count, memory, time and job name |
| `MODULES`, `PYTHON_BIN` | Compute-node environment and model interpreter |
| `MAX_JOBS` | Limit the number of training configurations; 0 uses the full grid |
| `MAX_CONCURRENT` | Maximum running tasks in each training/validation array |
| `TRAIN`, `TRAIN_GRID` | Fixed training arguments and hyperparameter choices |
| `VALIDATE`, `VALIDATE_GRID` | Validation workers and fusion-alpha choices |
| `SUMMARIZE` | Expected fold count, ranking length and incomplete-fold handling |
| `EXTRA_SBATCH_*` | Optional qos, constraint, reservation, nice or mail settings |
| `CLEAN_BEFORE_*` | Opt-in cleanup before submission; default false |

Resource settings become `sbatch` arguments inside the Slurm controller. This
avoids the Slurm limitation that shell variables are not expanded in `#SBATCH`
lines. Configure a CPU account/partition for `JOB_SUMMARIZE` if required on Helios.

A grid value uses `|` between choices. Any training CLI key can be a grid axis:

```bash
declare -A TRAIN_GRID=(
  [mri_model]='densenet121|resnet10|resnet18'
  [feature_groups]='clinical|clinical,kinetic,morphology,heterogeneity'
  [feature_model]='xgboost|mlp'
  [batch_size]='4|8'
  [lr]='1e-4|3e-4'
)
```

Feature groups use commas inside a choice. `target_size` uses three space-separated
integers. Strings such as `Patient ID` stay a single argument. Booleans use
`true`/`false`, which become the model's presence/absence flags. Additional model
constructor/Lightning Trainer options use the existing CLI's JSON object argument:

```bash
TRAIN[xgb_extra_json]='{"reg_alpha": 0.1, "reg_lambda": 2.0}'
TRAIN[trainer_extra_json]='{"precision": "16-mixed", "gradient_clip_val": 1.0}'
```

There is no JSON configuration parser in the controller; these strings are passed
unchanged to the existing model CLI. The model validates their contents at runtime.
Unknown configuration keys, empty grid choices, invalid booleans, invalid resource
counts, and out-of-range task indices fail before submission.

The shipped grid has **24 training jobs**, with up to **4 running concurrently**.
Validation uses the same 24 training rows. Three fusion-alpha choices would create
72 validation jobs. `MAX_JOBS` caps training rows before validation expansion.
Each array task runs all configured folds for its selected combination.

## Queued jobs and checkpoint matching

The controller saves resolved shell values and a frozen copy of itself under
`PATHS[slurm_logs]/run.*`. Workers verify the configuration snapshot's SHA-256 before
sourcing it. Editing the original configuration/controller after submission does
not change queued tasks. Preserve these run directories until all tasks finish.

Training and validation compute matching experiment IDs from all selected training
settings and resolved input/cache paths. Validation-only settings, resource limits,
and output paths do not affect IDs. Keep training settings and input paths the same
when validating. The new IDs differ from the old Python planner's IDs: **retrain
with this controller before validating its checkpoints**. Existing checkpoints are
retained because cleanup now defaults to false.

`--dry-run` prints the array size, resource command, cleanup targets (if enabled),
and task 0 command, without submitting, loading modules or changing files.
To inspect another row:

```bash
bash run_experiment.slurm --validate --task-index 23 --dry-run
bash run_experiment.sh --train --config /absolute/path/custom.conf --dry-run
```

Stages submit independently. Schedule validation/summary after successful jobs
with their printed Slurm IDs:

```bash
bash run_experiment.sh --validate --dependency afterok:12345
bash run_experiment.sh --summarize --dependency afterok:12346
```

Cleanup, when enabled, occurs once before submission. It rejects broad directories,
input/cache paths, overlapping output roots, and Slurm log/snapshot directories.
Do not enable cleanup while older jobs write to the same output directories.

## Other entry points

`ConfigurableImagingFeaturesFusionBinary.sh`,
`submit_configurable_imaging_features_fusion.sh`,
`validate_configurable_imaging_features_fusion.sh`, and
`summarize_configurable_imaging_features_fusion.sh` invoke the same controller with
the relevant stage. They read `experiment.conf`; they do not maintain separate grids.
Use `bash <launcher>.sh`, rather than `sbatch <launcher>.sh`.

The historical DenseNet, download, and older fusion launchers now only call `sbatch`
on a matching `.slurm` file. Their existing workloads/resources remain in those
`.slurm` files, including historical paths and module entry points. They do not use
the configurable fusion controller; ensure their log directories already exist.

Run the shell integration tests with:

```bash
bash tests/slurm_controller_test.slurm
```

Tests stub Slurm and the model executable; actual GPU training requires Helios and
the configured datasets/environment.

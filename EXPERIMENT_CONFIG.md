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
| `MODULES`, `PYTHON_BIN`, `PYTHON_VENV_ROOT` | Compute modules, interpreter, and optional architecture-specific virtual environments |
| `AUTO_INSTALL_PYTHON_PACKAGES`, `PYTHON_PACKAGES` | Automatic compute-node dependency setup and editable pip package list |
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

## Compute-node Python dependencies

The shipped configuration automatically prepares the required Python packages
before running training, validation or summary tasks:

```bash
PYTHON_VENV_ROOT=.venvs
AUTO_INSTALL_PYTHON_PACKAGES=true
PYTHON_PACKAGES=(
  monai==1.5.1 pytorch_lightning==2.5.6 lightning==2.5.6 torchmetrics==1.8.2
  requests pandas scikit-learn xgboost nibabel openpyxl tensorboard einops matplotlib
  scipy joblib filelock tqdm pillow
)
```

This combines the package lists from `denseBinary.slurm` and
`mriBreastDuke/run_xgboost_fusion.slurm`, including explicit dependencies imported
by the current pipeline. Add or pin packages in `PYTHON_PACKAGES`; pip resolves
their transitive dependencies. NumPy and CUDA-enabled PyTorch are supplied by
`MODULES`, checked before setup, and constrained to their module versions during
installation. An incompatible requested package fails rather than replacing
those versions.

GH200 nodes use aarch64 Python. Shared `~/.local` packages installed on an x86
login node can contain incompatible NumPy binaries. Every compute task keeps
`PYTHONNOUSERSITE=1` before and after module loading. Keep `PYTHONPATH` free of
manually added `~/.local` paths; cluster modules may need their own paths.

The controller creates a virtual environment with `--system-site-packages`
**inside the compute job**. It uses
`<PYTHON_VENV_ROOT>/<compute architecture>/runtime-<hash>/bin/python`.
The hash includes the base Python executable/version, module NumPy/PyTorch
metadata, configured modules, package list, and bootstrap revision. This keeps
different runtimes and package sets separate. Relative roots resolve against the
original configuration directory and are saved as absolute paths in snapshots.

The first task holds a `flock` lock while creating the environment and installing
packages; other tasks wait, then reuse it. Setup runs `pip check` and invokes all
three actual pipeline modules with `--help` to catch import failures before
experiments start. A ready marker is written only after these checks pass.
Failures stop the task; later tasks can retry incomplete setup. Completed
environments are reused without pip installs. First-use setup needs package-index
network access, writable storage, `venv`, `pip`, and the Linux `flock` utility.

With the updated configuration, submit normally from the login node:

```bash
bash run_experiment.sh --train --dry-run
bash run_experiment.sh --train
```

Dry runs show the package list and environment root without loading modules,
calling Python, installing packages, or creating environments. Training,
validation, and summary use the same setup. Resubmit failed jobs after updating:
already queued jobs retain their frozen controller and package configuration.

To use an environment you prepare manually on the target compute architecture,
set `AUTO_INSTALL_PYTHON_PACKAGES=false`. With a nonempty root, jobs then use
`<root>/$(uname -m)/bin/python` and fail if it is missing. With
`PYTHON_VENV_ROOT=''`, jobs use `PYTHON_BIN` and only the module's packages.
Existing custom configurations without the new install flag retain this manual
behavior; copy the new settings above to enable automatic setup.

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


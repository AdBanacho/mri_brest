# One Bash configuration for the MRI fusion workflow

Edit [`run_configuration.sh`](run_configuration.sh), then run from the repository root:

```bash
bash run_experiment.sh --train --dry-run
bash run_experiment.sh --train
bash run_experiment.sh --validate
bash run_experiment.sh --summarize
```

The configuration is a Bash file of `CFG_*` associative arrays. Each value is a
TOML literal: numbers and booleans can be written as `'30'` and `'false'`;
strings need double quotes inside the Bash quotes, such as `'"lasso"'`.
Grid values are TOML arrays, for example
`[mri_model]='["resnet10", "resnet18"]'`. To sweep feature groups, use
`[feature_groups]='[["clinical"], ["clinical", "kinetic"]]'`. Extra model
options can be set in `CFG_train_xgb_extra`, `CFG_train_mlp_extra`,
`CFG_train_mri_extra`, and `CFG_train_trainer_extra`; add arbitrary extra
sweeps as arrays of inline tables under `CFG_train_grid`. Run the file with
Bash; the shell on Helios must support associative arrays and namerefs
(Bash 4.3+).

The Bash launcher selects Python 3.10+ with `tomllib` or `tomli`; it tries
`python3.11` and `ML-bundle` if the default interpreter is too old. If those
are unavailable, inspect `module avail Python`, load an available version, or
set `PYTHON_BIN=/path/to/python3.11`. Compute jobs still load the modules
listed in `CFG_jobs`. Install the project dependencies in that environment
before submitting jobs. The Python planner remains responsible for expanding
the grid, IDs, cleanup checks, snapshots, and Slurm submission.

The dry run prints the array size, Slurm request, cleanup directories and the
first Python task command; it never submits or deletes anything. Submit training,
then validation after training completes, then summarization after validation.
The summary is a single CPU job. Long flags `--training` and `--validation`
also work. `CFG_jobs[max_jobs]=0` selects the full grid, and
`CFG_jobs[max_concurrent]` throttles the Slurm arrays.

The default cleanup flags delete checkpoints and TensorBoard logs once before
training, and validation outputs once before validation. Set
`CFG_cleanup[before_train]='false'` and
`CFG_cleanup[before_validate]='false'` to retain older results. The runner
rejects broad directories, configured input paths and overlapping outputs.
Each submission saves a resolved `experiment-<hash>.json` snapshot in the
configured Slurm log directory. Queued tasks use that snapshot, so changing the
Bash configuration after submission cannot change active jobs.

Paths may be absolute, start with `~`, use `@CONSTANT_NAME` from
`mriBreastDuke/constants.py`, or be relative to the Bash configuration file.
The metadata workbook may exist only on your cluster and must be accessible
from compute nodes. To use another configuration:

```bash
bash run_experiment.sh --config /path/to/my_configuration.sh --train --dry-run
```

A configuration can start as a copy of `run_configuration.sh`. The original
`experiment.toml` and direct `python3 run_experiment.py --config ...` entry
remain supported for existing experiments. The Bash file is the editable
configuration for new runs.

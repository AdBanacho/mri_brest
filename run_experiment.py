#!/usr/bin/env python3
"""One TOML configuration for the configurable fusion Slurm workflow.

Planning and submission use only the Python standard library. Heavy ML imports
occur inside individual Slurm tasks, after the requested environment is loaded.
"""

import os
import shlex
import shutil
import sys


def _use_supported_python():
    """Relaunch from old Helios login-node Python before importing this runner."""
    if sys.version_info >= (3, 10):
        return
    script = os.path.abspath(__file__)
    python311 = shutil.which("python3.11")
    if python311:
        os.execv(python311, [python311, script] + sys.argv[1:])
    sys.exit(
        "This project requires Python 3.10 or newer. "
        "Use bash run_experiment.sh for automatic Python selection, "
        "or load an available Python module (check: module spider Python)."
    )


_use_supported_python()

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from mriBreastDuke import constants


ROOT = Path(__file__).resolve().parent
MODULES = {
    "train": "mriBreastDuke.configurable_imaging_features_fusion_workflow",
    "validate": "mriBreastDuke.validate_configurable_imaging_features_fusion",
    "summarize": "mriBreastDuke.summarize_configurable_imaging_features_fusion",
}
TRAIN_KEYS = set("""
    mri_model subtraction_mode feature_groups feature_model feature_selector
    imaging_patient_id_column allow_missing_imaging_features include_sensitive
    use_annotation_boxes epoch num_folds threshold_calibration_folds batch_size
    num_workers positive_boost sensitivity_lambda lr xgb_n_estimators
    xgb_max_depth xgb_learning_rate xgb_subsample xgb_colsample_bytree
    xgb_n_jobs lasso_cv_folds lasso_cs lasso_max_iter lasso_tolerance
    lasso_min_features lasso_n_jobs lasso_plot_top_n mlp_hidden_layers
    mlp_alpha mlp_learning_rate mlp_max_iter xgb_extra mlp_extra
    mri_extra trainer_extra
""".split())
VALIDATE_KEYS = {"fusion_alpha", "num_workers"}
SUMMARY_KEYS = {"expected_folds", "top_n", "include_incomplete"}
PATH_KEYS = {
    "clinical_features_file": "clinical_features_file",
    "metadata_file": "metadata_file",
    "imaging_features_file": "imaging_features_file",
    "annotation_boxes_file": "annotation_boxes_file",
    "image_root": "image_root",
    "prepared_root": "prepared_root",
    "subtraction_root": "subtraction_root",
    "logs_root": "logs_root",
    "checkpoint_root": "checkpoint_root",
    "validation_root": "output_dir",
}
BOOL_KEYS = {"allow_missing_imaging_features", "include_sensitive", "use_annotation_boxes", "include_incomplete"}
EXTRA_KEYS = {"xgb_extra", "mlp_extra", "mri_extra", "trainer_extra"}
RESOURCE_KEYS = {"job_name", "cpus", "gpus", "memory", "time", "account", "partition", "modules", "extra_sbatch"}


def _keys(section: dict, allowed: set[str], label: str) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"[{label}] must be a table")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in [{label}]: {', '.join(unknown)}")


def _path(raw: str, config_dir: Path) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError("Paths must be nonempty strings")
    if raw.startswith("@"):
        name = raw[1:]
        if not name.isupper() or not hasattr(constants, name):
            raise ValueError(f"Unknown constant reference: {raw}")
        raw = getattr(constants, name)
        if not isinstance(raw, (str, Path)):
            raise ValueError(f"Constant {name} is not a path")
    path = Path(raw).expanduser()
    return str((config_dir / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path: Path, config_base: Path = None) -> dict:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _keys(config, {"data", "paths", "jobs", "train", "validate", "summarize", "cleanup"}, "root")
    cleanup = config.get("cleanup", {})
    _keys(cleanup, {"before_train", "before_validate"}, "cleanup")
    for key, value in cleanup.items():
        if type(value) is not bool:
            raise ValueError(f"[cleanup].{key} must be true or false")
    data = config.get("data", {})
    _keys(data, {"seed", "target_size"}, "data")
    size = data.get("target_size", [256, 256, 64])
    if not isinstance(size, list) or len(size) != 3 or any(type(n) is not int or n <= 0 for n in size):
        raise ValueError("[data].target_size must be three positive integers")
    if type(data.get("seed", 42)) is not int:
        raise ValueError("[data].seed must be an integer")
    paths = config.get("paths", {})
    _keys(paths, set(PATH_KEYS) | {"summary_root", "slurm_logs"}, "paths")
    config["paths"] = {key: _path(value, config_base or path.parent) for key, value in paths.items()}
    train = config.get("train", {})
    _keys(train, TRAIN_KEYS | {"grid"}, "train")
    _keys(train.get("grid", {}), TRAIN_KEYS, "train.grid")
    if not train:
        raise ValueError("A [train] table is required")
    validate = config.get("validate", {})
    _keys(validate, VALIDATE_KEYS | {"grid"}, "validate")
    _keys(validate.get("grid", {}), VALIDATE_KEYS, "validate.grid")
    _keys(config.get("summarize", {}), SUMMARY_KEYS, "summarize")
    jobs = config.get("jobs", {})
    _keys(jobs, RESOURCE_KEYS | {"max_jobs", "max_concurrent", "train", "validate", "summarize"}, "jobs")
    for stage in MODULES:
        _keys(jobs.get(stage, {}), RESOURCE_KEYS, f"jobs.{stage}")
    max_jobs = jobs.get("max_jobs", 0)
    concurrent = jobs.get("max_concurrent", 8)
    if type(max_jobs) is not int or max_jobs < 0 or type(concurrent) is not int or concurrent < 1:
        raise ValueError("jobs.max_jobs must be >= 0 and jobs.max_concurrent must be >= 1")
    config["data"] = data
    config["train"] = train
    config["validate"] = validate
    config["jobs"] = jobs
    config["summarize"] = config.get("summarize", {})
    config["cleanup"] = cleanup
    return config


def _grid(base: dict, allowed: set[str], label: str) -> list[dict]:
    fixed = {key: value for key, value in base.items() if key != "grid"}
    axes = base.get("grid", {})
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"[{label}.grid].{key} needs a nonempty array")
    keys = sorted(axes)
    result = []
    for values in itertools.product(*(axes[key] for key in keys)):
        options = {**fixed, **dict(zip(keys, values))}
        _keys(options, allowed, label)
        if "feature_groups" in options and (not isinstance(options["feature_groups"], list) or not options["feature_groups"]):
            raise ValueError("feature_groups must be a nonempty array; each grid choice is an array")
        for key in EXTRA_KEYS & options.keys():
            if not isinstance(options[key], dict):
                raise ValueError(f"{key} must be a TOML table/inline table")
        result.append(options)
    return result


def _train_configs(config: dict) -> list[dict]:
    rows = _grid(config["train"], TRAIN_KEYS, "train")
    cap = config["jobs"].get("max_jobs", 0)
    rows = rows[:cap] if cap else rows
    ids = [_experiment_id(config, row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("The training grid contains duplicate configurations")
    return rows


def _validation_configs(config: dict) -> list[tuple[dict, dict]]:
    choices = _grid(config["validate"], VALIDATE_KEYS, "validate")
    return list(itertools.product(_train_configs(config), choices))


def _experiment_id(config: dict, training: dict) -> str:
    # Include data and input locations: a changed cohort must not reuse checkpoints.
    identity = {"train": training, "data": config["data"], "paths": {
        key: value for key, value in config["paths"].items()
        if key not in {"logs_root", "checkpoint_root", "validation_root", "summary_root", "slurm_logs"}
    }}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "cfg" + hashlib.sha256(encoded).hexdigest()[:12]


def _options(values: dict) -> list[str]:
    args = []
    for key, value in sorted(values.items()):
        key = key + "_json" if key in EXTRA_KEYS else key
        if isinstance(value, bool):
            if value:
                args.append("--" + key)
        elif isinstance(value, list):
            args.extend(["--" + key, *(str(item) for item in value)])
        elif isinstance(value, dict):
            args.extend(["--" + key, json.dumps(value, sort_keys=True)])
        else:
            args.extend(["--" + key, str(value)])
    return args


def task_command(config: dict, stage: str, index: int) -> list[str]:
    paths = config["paths"]
    if stage == "summarize":
        if index != 0:
            raise ValueError("Summarize has one task (index 0)")
        summary = dict(config["summarize"])
        summary.setdefault("expected_folds", config["train"].get("num_folds", 5))
        options = {**summary, "input_dir": paths.get("validation_root", str(ROOT / "validation_charts"))}
        if "summary_root" in paths:
            options["output_dir"] = paths["summary_root"]
    else:
        rows = _train_configs(config) if stage == "train" else _validation_configs(config)
        if index < 0 or index >= len(rows):
            raise ValueError(f"Task index {index} is outside 0..{len(rows) - 1}")
        training, validation = (rows[index], {}) if stage == "train" else rows[index]
        options = {**training, "seed": config["data"].get("seed", 42),
                   "target_size": config["data"].get("target_size", [256, 256, 64]),
                   "experiment_id": _experiment_id(config, training)}
        for path_key, arg_key in PATH_KEYS.items():
            if path_key in paths and (stage == "train" or arg_key not in {"logs_root", "checkpoint_root"}):
                if stage == "train" and arg_key == "output_dir":
                    continue
                options[arg_key] = paths[path_key]
        if stage == "validate":
            options = {key: value for key, value in options.items() if key not in {
                "epoch", "lasso_plot_top_n", "logs_root", "xgb_n_estimators",
                "xgb_max_depth", "xgb_learning_rate", "xgb_subsample",
                "xgb_colsample_bytree", "xgb_n_jobs", "mlp_hidden_layers",
                "mlp_alpha", "mlp_learning_rate", "mlp_max_iter", "xgb_extra",
                "mlp_extra", "trainer_extra"
            }}
            options.update(validation)
            options.setdefault("fusion_alpha", 0.5)
            options.setdefault("checkpoint_root", paths.get("checkpoint_root", constants.CHECKPOINTS_PATH))
    if stage == "summarize":
        return [sys.executable, "-m", MODULES[stage], *(
            item.replace("_", "-") if item.startswith("--") else item
            for item in _options(options)
        )]
    return [sys.executable, "-m", MODULES[stage], *_options(options)]


def _resources(config: dict, stage: str) -> dict:
    jobs = config["jobs"]
    result = {key: value for key, value in jobs.items() if key in RESOURCE_KEYS}
    result.update(jobs.get(stage, {}))
    result.setdefault("job_name", "mri_" + stage)
    result.setdefault("cpus", 2 if stage == "summarize" else 16)
    result.setdefault("gpus", 0 if stage == "summarize" else 1)
    result.setdefault("memory", "8G" if stage == "summarize" else "48G")
    result.setdefault("time", "02:00:00" if stage == "summarize" else "24:00:00")
    result.setdefault("modules", ["ML-bundle"])
    for key in ("cpus", "gpus"):
        if type(result[key]) is not int or result[key] < (1 if key == "cpus" else 0):
            raise ValueError(f"jobs.{stage}.{key} has an invalid value")
    for key in ("modules", "extra_sbatch"):
        if key in result and (not isinstance(result[key], list) or not all(isinstance(s, str) for s in result[key])):
            raise ValueError(f"jobs.{stage}.{key} must be an array of strings")
    return result


def _cleanup_targets(config: dict, stage: str) -> list[Path]:
    enabled = config.get("cleanup", {}).get(
        "before_train" if stage == "train" else "before_validate", False
    )
    if stage == "summarize" or not enabled:
        return []
    paths = config["paths"]
    names = ("checkpoint_root", "logs_root") if stage == "train" else ("validation_root",)
    missing = [name for name in names if name not in paths]
    if missing:
        raise ValueError(f"Cleanup requires explicit [paths] settings: {', '.join(missing)}")
    targets = [Path(paths[name]).resolve() for name in names]
    if stage == "validate" and "summary_root" in paths:
        summary = Path(paths["summary_root"]).resolve()
        if summary not in targets[0].parents and summary != targets[0] and targets[0] not in summary.parents:
            targets.append(summary)
        elif summary in targets[0].parents:
            raise ValueError("summary_root cannot contain validation_root when cleanup is enabled")

    protected_names = ("clinical_features_file", "metadata_file", "imaging_features_file",
                       "annotation_boxes_file", "image_root", "prepared_root", "subtraction_root")
    protected = [Path(paths[key]).resolve() for key in protected_names if key in paths]
    slurm_logs = Path(paths.get("slurm_logs", str(ROOT / "logs"))).resolve()
    outputs = [Path(paths[key]).resolve() for key in
               ("checkpoint_root", "logs_root", "validation_root") if key in paths]
    for target in targets:
        if (len(target.parts) < 4 or target == ROOT or target in ROOT.parents
                or target == Path.home() or target in Path.home().parents):
            raise ValueError(f"Refusing to clean a broad or protected directory: {target}")
        if any(target == item or target in item.parents for item in protected):
            raise ValueError(f"Cleanup directory contains a configured input: {target}")
        if target == slurm_logs or target in slurm_logs.parents or slurm_logs in target.parents:
            raise ValueError(f"Cleanup directory overlaps Slurm logs/snapshots: {target}")
        if any(target != item and (target in item.parents or item in target.parents)
               for item in outputs):
            raise ValueError(f"Cleanup directory overlaps another output root: {target}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Cleanup target is not a directory: {target}")
    if len(targets) > 1 and any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(targets) for right in targets[index + 1:]
    ):
        raise ValueError("Cleanup directories must not overlap")
    return targets


def submit(config: dict, config_path: Path, stage: str, dry_run: bool) -> None:
    count = (1 if stage == "summarize" else
             len(_train_configs(config)) if stage == "train" else len(_validation_configs(config)))
    if not count:
        raise ValueError("The job grid is empty")
    resources = _resources(config, stage)
    cleanup_targets = _cleanup_targets(config, stage)
    logs = Path(config["paths"].get("slurm_logs", str(ROOT / "logs")))
    snapshot = json.dumps(config, sort_keys=True, indent=2) + "\n"
    snapshot_hash = hashlib.sha256(snapshot.encode()).hexdigest()[:16]
    snapshot_path = logs / f"experiment-{snapshot_hash}.json"
    script = ["#!/bin/bash -l", "set -euo pipefail", "cd " + shlex.quote(str(ROOT))]
    for module in resources["modules"]:
        if not module or any(c in module for c in "\n\r;`$()"):
            raise ValueError("Module names must be simple module identifiers")
        script.append("module load " + shlex.quote(module))
    script.append(shlex.join(["python3", str(ROOT / "run_experiment.py"),
                              "--snapshot", str(snapshot_path), "--" + stage,
                              "--task-index"]) + ' "${SLURM_ARRAY_TASK_ID:-0}"')
    command = ["sbatch", "--parsable", "--job-name=" + str(resources["job_name"]),
               "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(resources["cpus"]),
               "--mem=" + str(resources["memory"]), "--time=" + str(resources["time"]),
               "--output=" + str(logs / "%x-%A_%a.out"),
               "--error=" + str(logs / "%x-%A_%a.err")]
    if resources["gpus"]:
        command.append("--gres=gpu:" + str(resources["gpus"]))
    for key in ("account", "partition"):
        if resources.get(key):
            command.append("--" + key + "=" + str(resources[key]))
    command.extend(resources.get("extra_sbatch", []))
    if stage != "summarize":
        command.append(f"--array=0-{count - 1}%{config['jobs'].get('max_concurrent', 8)}")
    if dry_run:
        print(f"{stage}: {count} job(s), max concurrent {config['jobs'].get('max_concurrent', 8)}")
        for target in cleanup_targets:
            print("Would remove before submission:", target)
        print("Submission:", shlex.join(command), "<generated-script>")
        print("Example task 0:", shlex.join(task_command(config, stage, 0)))
        return
    logs.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if snapshot_path.read_text(encoding="utf-8") != snapshot:
            raise ValueError(f"Snapshot hash collision at {snapshot_path}")
    else:
        snapshot_path.write_text(snapshot, encoding="utf-8")
    for target in cleanup_targets:
        if target.exists():
            print(f"Removing previous {stage} output: {target}", flush=True)
            shutil.rmtree(target)
    completed = subprocess.run(command, input="\n".join(script) + "\n", text=True, check=True, capture_output=True)
    print(f"Submitted {stage}: {count} job(s), Slurm ID {completed.stdout.strip()}")
    print(f"Resolved configuration: {snapshot_path}")


def load_snapshot(path: Path) -> dict:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()[:16]
    if path.name != f"experiment-{actual}.json":
        raise ValueError(f"Configuration snapshot was changed: {path}")
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, default=ROOT / "experiment.toml")
    source.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config-base", type=Path, help=argparse.SUPPRESS)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--train", "--training", dest="stage", action="store_const", const="train")
    choice.add_argument("--validate", "--validation", dest="stage", action="store_const", const="validate")
    choice.add_argument("--summarize", dest="stage", action="store_const", const="summarize")
    parser.add_argument("--dry-run", action="store_true", help="Show the grid and command without submitting")
    parser.add_argument("--task-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config_path = (args.snapshot or args.config).expanduser().resolve()
        config = (load_snapshot(config_path)
                  if args.snapshot else load_config(config_path, args.config_base))
        if args.task_index is not None:
            command = task_command(config, args.stage, args.task_index)
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
        else:
            submit(config, config_path, args.stage, args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

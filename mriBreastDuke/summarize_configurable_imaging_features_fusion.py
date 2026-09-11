"""Summarize all configurable imaging-feature fusion validation runs.

The validation Slurm array writes one ``validation_metrics.csv`` per
configuration below ``validation_charts``.  This module discovers those files
and creates a cross-configuration report in ``validation_charts/summary``.
Incomplete configurations remain visible in the CSV outputs, but are excluded
from the ranking by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auc_roc",
    "decision_threshold",
)
RANKING_COLUMNS = (
    "fusion_auc_roc_mean",
    "fusion_balanced_accuracy_mean",
    "fusion_sensitivity_mean",
    "fusion_specificity_mean",
)
CONFIG_COLUMNS = (
    "mri_model",
    "subtraction_mode",
    "feature_groups",
    "feature_model",
    "feature_selector",
    "batch_size",
    "positive_boost",
    "sensitivity_lambda",
    "lr",
    "fusion_alpha",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create cross-experiment CSV summaries and comparison charts from "
            "configurable imaging-feature fusion validation outputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="validation_charts",
        help="Directory containing per-configuration validation results.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report directory (default: <input-dir>/summary).",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=5,
        help="Fallback expected fold count when validation_config.json is absent.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of ranked configurations shown in each chart.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete configurations in the ranking and charts.",
    )
    return parser.parse_args()


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "validation_config.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read validation configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Validation configuration must contain an object: {path}")
    return {key: _json_value(item) for key, item in value.items()}


def _canonicalize_tabular_metrics(
    metrics: pd.DataFrame,
    feature_model: str | None,
) -> pd.DataFrame:
    output = metrics.copy()
    if not feature_model:
        return output
    for metric in CLASSIFICATION_METRICS:
        source = f"{feature_model}_{metric}"
        if source in output.columns:
            output[f"tabular_{metric}"] = output[source]
    return output


def _configuration_label(row: pd.Series) -> str:
    groups = str(row.get("feature_groups", "?"))
    if len(groups) > 34:
        groups = groups[:31] + "..."
    return (
        f"{row.get('mri_model', '?')} | {row.get('subtraction_mode', '?')} | "
        f"{groups} | {row.get('feature_model', '?')} | "
        f"boost={row.get('positive_boost', '?')}"
    )


def _discover_runs(input_dir: Path) -> list[tuple[Path, Path]]:
    summary_dir = input_dir / "summary"
    discovered = []
    for metrics_path in sorted(input_dir.rglob("validation_metrics.csv")):
        if summary_dir == metrics_path.parent or summary_dir in metrics_path.parents:
            continue
        discovered.append((metrics_path.parent, metrics_path))
    return discovered


def collect_results(
    input_dir: Path,
    fallback_expected_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Collect fold rows and configuration summaries from all completed files."""
    if fallback_expected_folds < 2:
        raise ValueError("expected_folds must be at least 2.")

    fold_frames: list[pd.DataFrame] = []
    configuration_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    runs = _discover_runs(input_dir)
    if not runs:
        raise FileNotFoundError(
            f"No validation_metrics.csv files were found below {input_dir}"
        )

    for run_dir, metrics_path in runs:
        try:
            metrics = pd.read_csv(metrics_path)
            if metrics.empty:
                warnings.append(f"Skipped empty metrics file: {metrics_path}")
                continue
            if "fold" not in metrics.columns:
                warnings.append(f"Skipped metrics file without a fold column: {metrics_path}")
                continue
            config = _read_config(run_dir)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            warnings.append(f"Skipped unreadable run {run_dir}: {error}")
            continue

        feature_model = str(config.get("feature_model", "")) or None
        metrics = _canonicalize_tabular_metrics(metrics, feature_model)
        run_id = str(run_dir.relative_to(input_dir))
        metrics.insert(0, "run_id", run_id)
        for key, value in config.items():
            if key not in metrics.columns:
                metrics[key] = value
        fold_frames.append(metrics)

        expected_folds = int(config.get("num_folds", fallback_expected_folds))
        numeric_folds = pd.to_numeric(metrics["fold"], errors="coerce").dropna()
        observed_folds = sorted({int(value) for value in numeric_folds})
        expected_fold_ids = list(range(1, expected_folds + 1))
        complete = observed_folds == expected_fold_ids
        row: dict[str, Any] = {
            "run_id": run_id,
            **{key: config.get(key) for key in CONFIG_COLUMNS},
            "expected_folds": expected_folds,
            "completed_folds": len(observed_folds),
            "observed_fold_ids": ",".join(map(str, observed_folds)),
            "is_complete": complete,
        }

        numeric = metrics.select_dtypes(include="number")
        for column in numeric.columns:
            if column == "fold":
                continue
            values = numeric[column].dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        configuration_rows.append(row)

    if not fold_frames:
        details = "\n".join(warnings)
        raise RuntimeError(f"No readable validation results found below {input_dir}\n{details}")

    return (
        pd.concat(fold_frames, ignore_index=True, sort=False),
        pd.DataFrame(configuration_rows),
        warnings,
    )


def rank_configurations(
    configurations: pd.DataFrame,
    include_incomplete: bool,
) -> pd.DataFrame:
    candidates = configurations.copy()
    if not include_incomplete:
        candidates = candidates[candidates["is_complete"]].copy()

    sort_columns = [column for column in RANKING_COLUMNS if column in candidates]
    if not sort_columns:
        raise ValueError(
            "Cannot rank configurations because no fusion performance metrics were found."
        )
    candidates = candidates.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    ).reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return candidates


def _save_metric_chart(ranked: pd.DataFrame, output_path: Path, top_n: int) -> None:
    metric_specs = (
        ("fusion_auc_roc_mean", "Fusion AUC ROC"),
        ("fusion_balanced_accuracy_mean", "Fusion balanced accuracy"),
        ("fusion_sensitivity_mean", "Fusion sensitivity"),
        ("fusion_specificity_mean", "Fusion specificity"),
    )
    available = [(column, label) for column, label in metric_specs if column in ranked]
    if not available or ranked.empty:
        return

    shown = ranked.head(top_n).iloc[::-1]
    y = np.arange(len(shown))
    height = 0.8 / len(available)
    fig_height = max(5.0, 0.55 * len(shown) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    for index, (column, label) in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * height
        ax.barh(y + offset, shown[column], height=height, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([_configuration_label(row) for _, row in shown.iterrows()])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean outer-fold score")
    ax.set_title("Top configurable fusion validation results")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_branch_auc_chart(ranked: pd.DataFrame, output_path: Path, top_n: int) -> None:
    specs = (
        ("image_auc_roc_mean", "MRI"),
        ("tabular_auc_roc_mean", "Tabular"),
        ("fusion_auc_roc_mean", "Fusion"),
    )
    available = [(column, label) for column, label in specs if column in ranked]
    if len(available) < 2 or ranked.empty:
        return

    shown = ranked.head(top_n).iloc[::-1]
    y = np.arange(len(shown))
    height = 0.8 / len(available)
    fig_height = max(5.0, 0.55 * len(shown) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    for index, (column, label) in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * height
        ax.barh(y + offset, shown[column], height=height, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([_configuration_label(row) for _, row in shown.iterrows()])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean outer-fold AUC ROC")
    ax.set_title("MRI, tabular, and fusion branch comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    configurations: pd.DataFrame,
    ranked: pd.DataFrame,
    warnings: list[str],
) -> None:
    completed = int(configurations["is_complete"].sum())
    lines = [
        "# Configurable fusion validation report",
        "",
        f"- Discovered configurations: {len(configurations)}",
        f"- Complete configurations: {completed}",
        f"- Incomplete configurations: {len(configurations) - completed}",
    ]
    if not ranked.empty:
        best = ranked.iloc[0]
        lines.extend(
            [
                f"- Best configuration: `{best['run_id']}`",
                f"- Best mean fusion AUC ROC: {best.get('fusion_auc_roc_mean', float('nan')):.4f}",
            ]
        )
    if warnings:
        lines.extend(["", "## Warnings", ""] + [f"- {item}" for item in warnings])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_summary(args: argparse.Namespace) -> Path:
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir) if args.output_dir else input_dir / "summary"
    if args.top_n < 1:
        raise ValueError("top_n must be at least 1.")
    output_dir.mkdir(parents=True, exist_ok=True)

    folds, configurations, warnings = collect_results(
        input_dir,
        fallback_expected_folds=args.expected_folds,
    )
    ranked = rank_configurations(configurations, args.include_incomplete)

    folds.to_csv(output_dir / "all_validation_metrics.csv", index=False)
    configurations.sort_values("run_id").to_csv(
        output_dir / "configuration_summary.csv",
        index=False,
    )
    ranked.to_csv(output_dir / "ranked_configurations.csv", index=False)
    if not ranked.empty:
        best = {key: _json_safe(value) for key, value in ranked.iloc[0].items()}
        (output_dir / "best_configuration.json").write_text(
            json.dumps(best, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _save_metric_chart(ranked, output_dir / "fusion_metric_comparison.png", args.top_n)
    _save_branch_auc_chart(ranked, output_dir / "branch_auc_comparison.png", args.top_n)
    _write_report(output_dir / "validation_report.md", configurations, ranked, warnings)

    print(f"Discovered {len(configurations)} validation configurations.")
    print(f"Complete configurations: {int(configurations['is_complete'].sum())}")
    print(f"Summary written to: {output_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see validation_report.md)")
    return output_dir


def main() -> None:
    generate_summary(parse_args())


if __name__ == "__main__":
    main()

"""Validate saved configurable MRI + tabular fusion checkpoints.

This module reconstructs the same patient-grouped folds as
``configurable_imaging_features_fusion_workflow``. It loads the saved MRI,
preprocessor, and XGBoost/MLP artifacts, performs one MRI inference pass per
fold, and reports image, tabular, and decision-fusion metrics.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

from mriBreastDuke.classificators import (
    NiftiClassifier,
    aligned_predict_proba,
    fuse_probabilities,
    probability_metrics,
    save_fusion_predictions,
)
from mriBreastDuke.configurable_imaging_features_fusion_workflow import (
    FEATURE_GROUPS,
    FEATURE_MODELS,
    FEATURE_SELECTORS,
    MRI_MODELS,
    build_experiment_name,
    make_mri_network,
    prepare_studies_and_features,
)
from mriBreastDuke.constants import (
    CHECKPOINTS_PATH,
    IMAGING_FEATURES_FILE_NAME,
    NIFTI_PATH,
    SEED,
    VALIDATION_CHART_PATH,
)
from mriBreastDuke.dataLoaders import (
    NiftiDataModule,
    SUBTRACTION_MODES,
    SUBTRACTION_NONE,
    get_input_channels,
)


BEST_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    r"val_sensitivity=(?P<sensitivity>[-+]?\d*\.?\d+)-"
    r"val_auc_roc=(?P<auc>[-+]?\d*\.?\d+)"
    r"(?:-v\d+)?\.ckpt$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate saved checkpoints from the configurable MRI + "
            "Imaging_Features.xlsx fusion workflow."
        )
    )
    parser.add_argument("--mri_model", choices=MRI_MODELS, default="densenet121")
    parser.add_argument(
        "--subtraction_mode",
        choices=SUBTRACTION_MODES,
        default=SUBTRACTION_NONE,
    )
    parser.add_argument(
        "--feature_groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=["clinical"],
    )
    parser.add_argument(
        "--feature_model",
        choices=FEATURE_MODELS,
        default="xgboost",
    )
    parser.add_argument(
        "--feature_selector",
        choices=FEATURE_SELECTORS,
        default="none",
    )
    parser.add_argument(
        "--imaging_features_file",
        default=IMAGING_FEATURES_FILE_NAME,
        help="Path to Imaging_Features.xlsx or its CSV export.",
    )
    parser.add_argument("--imaging_patient_id_column", default="Patient ID")
    parser.add_argument("--allow_missing_imaging_features", action="store_true")
    parser.add_argument("--include_sensitive", action="store_true")
    parser.add_argument("--lasso_cv_folds", type=int, default=5)
    parser.add_argument("--lasso_cs", type=int, default=20)
    parser.add_argument("--lasso_max_iter", type=int, default=5000)
    parser.add_argument("--lasso_tolerance", type=float, default=1e-4)
    parser.add_argument("--lasso_min_features", type=int, default=1)
    parser.add_argument("--lasso_n_jobs", type=int, default=8)

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--positive_boost", type=float, default=1.0)
    parser.add_argument("--sensitivity_lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="MRI probability weight; the tabular weight is 1 - alpha.",
    )
    parser.add_argument(
        "--checkpoint_root",
        default=CHECKPOINTS_PATH,
        help="Root containing <experiment>/fold_N/checkpoints directories.",
    )
    parser.add_argument(
        "--output_dir",
        default=VALIDATION_CHART_PATH,
        help="Root for validation metrics, predictions, and charts.",
    )
    return parser.parse_args()


def _resolve_directory(path_like, create=False):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _validate_inputs(studies, num_folds, fusion_alpha):
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1.")
    for column in ("label", "patientId"):
        if column not in studies.columns:
            raise ValueError(f"Validation data is missing required column: {column}")
        if studies[column].isna().any():
            raise ValueError(f"Validation column contains missing values: {column}")

    labels = studies["label"].to_numpy()
    integer_labels = labels.astype(np.int64)
    unique_labels = np.unique(integer_labels)
    if not np.array_equal(labels, integer_labels):
        raise ValueError("Labels must be integers encoded from 0 to C-1.")
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        raise ValueError("Labels must be contiguous integers encoded from 0 to C-1.")

    group_label_counts = studies.groupby("patientId", sort=False)["label"].nunique()
    if (group_label_counts > 1).any():
        raise ValueError("Each patientId must have exactly one label.")
    groups_per_class = (
        studies[["patientId", "label"]]
        .drop_duplicates()
        .groupby("label")["patientId"]
        .nunique()
        .reindex(unique_labels, fill_value=0)
        .to_numpy()
    )
    if np.any(groups_per_class < num_folds):
        raise ValueError(
            "Every class must contain at least num_folds distinct patients; "
            f"counts={groups_per_class.tolist()}, num_folds={num_folds}."
        )
    return integer_labels, len(unique_labels)


def _checkpoint_score(path):
    match = BEST_CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return (0, float("-inf"), float("-inf"), -1, path.stat().st_mtime)
    return (
        1,
        float(match.group("sensitivity")),
        float(match.group("auc")),
        int(match.group("epoch")),
        path.stat().st_mtime,
    )


def find_best_checkpoint(checkpoint_dir):
    candidates = list(checkpoint_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best-*.ckpt checkpoint found in: {checkpoint_dir}"
        )
    return max(candidates, key=_checkpoint_score)


def _compute_class_weights(labels, num_classes, positive_boost):
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=num_classes)
    total = class_counts.sum().float()
    weights = total / (num_classes * class_counts.float().clamp_min(1.0))
    if num_classes == 2:
        weights[1] *= positive_boost
    return weights


def _load_image_model(args, checkpoint_path, num_classes, class_weights, device):
    network = make_mri_network(
        args.mri_model,
        input_channels=get_input_channels(args.subtraction_mode),
        num_classes=num_classes,
    )
    model = NiftiClassifier(
        network,
        num_classes=num_classes,
        lr=args.lr,
        class_weights=class_weights,
        sensitivity_lambda=args.sensitivity_lambda,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise RuntimeError(
            f"Checkpoint does not contain a model state_dict: {checkpoint_path}"
        )
    model.load_state_dict(state_dict, strict=True)
    del state_dict, checkpoint
    return model.to(device).eval()


def _predict_image_probabilities(model, dataloader, num_classes, device):
    probabilities = []
    labels = []
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for images, batch_labels in dataloader:
            logits = model(images.to(device, non_blocking=non_blocking))
            if num_classes == 2 and (logits.ndim == 1 or logits.shape[1] == 1):
                positive = torch.sigmoid(logits.view(-1))
                batch_probabilities = torch.stack((1.0 - positive, positive), dim=1)
            else:
                batch_probabilities = torch.softmax(logits, dim=1)
            probabilities.append(batch_probabilities.detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())

    if not probabilities:
        raise RuntimeError("The validation DataLoader produced no batches.")
    return np.concatenate(probabilities), np.concatenate(labels)


def _save_confusion_matrix(labels, probabilities, output_path, title, normalize=False):
    num_classes = probabilities.shape[1]
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(num_classes),
        normalize="true" if normalize else None,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(matrix, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    threshold = matrix.max() * 0.6 if matrix.size else 0
    for row in range(num_classes):
        for column in range(num_classes):
            value = matrix[row, column]
            label = f"{value:.2f}" if normalize else str(int(value))
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(labels, probabilities, output_path, title):
    num_classes = probabilities.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False

    if num_classes == 2:
        if np.unique(labels).size == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            ax.plot(fpr, tpr, label=f"class_1 (AUC={auc(fpr, tpr):.3f})")
            plotted = True
    else:
        one_hot = label_binarize(labels, classes=np.arange(num_classes))
        for class_index in range(num_classes):
            if np.unique(one_hot[:, class_index]).size < 2:
                continue
            fpr, tpr, _ = roc_curve(
                one_hot[:, class_index],
                probabilities[:, class_index],
            )
            ax.plot(
                fpr,
                tpr,
                label=f"class_{class_index} (AUC={auc(fpr, tpr):.3f})",
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print(f"[ROC] Skipping {output_path}: targets contain one class.", flush=True)
        return

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_branch_charts(labels, probabilities, output_dir, title):
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(
        labels,
        probabilities,
        output_dir / "confusion_matrix.png",
        title=f"{title} Confusion Matrix",
    )
    _save_confusion_matrix(
        labels,
        probabilities,
        output_dir / "confusion_matrix_normalized.png",
        title=f"{title} Normalized Confusion Matrix",
        normalize=True,
    )
    _save_roc_curve(
        labels,
        probabilities,
        output_dir / "roc_curve.png",
        title=f"{title} ROC Curve",
    )


def run_validation(args):
    studies, _, _, selected_groups = prepare_studies_and_features(args)
    labels, num_classes = _validate_inputs(
        studies,
        num_folds=args.num_folds,
        fusion_alpha=args.fusion_alpha,
    )
    experiment_name = build_experiment_name(args, selected_groups)
    checkpoint_root = _resolve_directory(args.checkpoint_root)
    output_root = _resolve_directory(
        Path(args.output_dir)
        / experiment_name
        / f"fusion_alpha_{args.fusion_alpha:g}",
        create=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[VALIDATION] Experiment: {experiment_name}", flush=True)
    print(f"[VALIDATION] Checkpoint root: {checkpoint_root}", flush=True)
    print(f"[VALIDATION] Output root: {output_root}", flush=True)
    print(f"[VALIDATION] Device: {device}", flush=True)

    splitter = StratifiedGroupKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=SEED,
    )
    groups = studies["patientId"].to_numpy()
    metrics_per_fold = []
    pooled = {
        "image": {"labels": [], "probabilities": []},
        args.feature_model: {"labels": [], "probabilities": []},
        "fusion": {"labels": [], "probabilities": []},
    }

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(studies, labels, groups),
        start=1,
    ):
        print(f"\n========== Validation fold {fold}/{args.num_folds} ==========", flush=True)
        train_df = studies.iloc[train_idx].reset_index(drop=True)
        val_df = studies.iloc[val_idx].reset_index(drop=True)
        overlap = set(train_df["patientId"]).intersection(val_df["patientId"])
        if overlap:
            raise RuntimeError(f"Patient leakage detected in fold {fold}.")

        checkpoint_dir = (
            checkpoint_root / experiment_name / f"fold_{fold}" / "checkpoints"
        )
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        preprocessor_path = checkpoint_dir / "tabular_preprocessor.joblib"
        tabular_model_path = checkpoint_dir / f"{args.feature_model}_model.joblib"
        artifact_paths = [preprocessor_path, tabular_model_path]
        selector_path = checkpoint_dir / "tabular_feature_selector.joblib"
        if args.feature_selector == "lasso":
            artifact_paths.append(selector_path)
        for artifact_path in artifact_paths:
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing saved fold artifact: {artifact_path}")

        class_weights = _compute_class_weights(
            train_df["label"].to_numpy(dtype=np.int64),
            num_classes=num_classes,
            positive_boost=args.positive_boost,
        )
        image_model = _load_image_model(
            args,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            class_weights=class_weights,
            device=device,
        )
        datamodule = NiftiDataModule(
            train_df=val_df,
            val_df=val_df,
            target_size=(256, 256, 64),
            image_root=NIFTI_PATH,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subtraction_mode=args.subtraction_mode,
        )
        datamodule.setup(stage="validate")
        image_probabilities, validation_labels = _predict_image_probabilities(
            image_model,
            datamodule.val_dataloader(),
            num_classes=num_classes,
            device=device,
        )
        expected_labels = val_df["label"].to_numpy(dtype=np.int64)
        if not np.array_equal(validation_labels, expected_labels):
            raise RuntimeError(
                "MRI predictions are not aligned with the tabular validation rows."
            )

        preprocessor = joblib.load(preprocessor_path)
        tabular_model = joblib.load(tabular_model_path)
        tabular_features = preprocessor.transform(val_df)
        tabular_feature_selector = None
        if args.feature_selector == "lasso":
            tabular_feature_selector = joblib.load(selector_path)
            tabular_features = tabular_feature_selector.transform(tabular_features)
        tabular_probabilities = aligned_predict_proba(
            tabular_model,
            tabular_features,
            num_classes=num_classes,
        )
        fused_probabilities = fuse_probabilities(
            image_probabilities,
            tabular_probabilities,
            alpha=args.fusion_alpha,
        )

        fold_metrics = {
            "fold": fold,
            "checkpoint_path": str(checkpoint_path),
        }
        fold_metrics.update(
            probability_metrics(validation_labels, image_probabilities, "image")
        )
        fold_metrics.update(
            probability_metrics(
                validation_labels,
                tabular_probabilities,
                args.feature_model,
            )
        )
        fold_metrics.update(
            probability_metrics(validation_labels, fused_probabilities, "fusion")
        )
        metrics_per_fold.append(fold_metrics)

        fold_output = output_root / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        save_fusion_predictions(
            val_df,
            validation_labels,
            image_probabilities,
            tabular_probabilities,
            fused_probabilities,
            fold_output / "fusion_validation_predictions.csv",
            tabular_model_name=args.feature_model,
        )
        branch_probabilities = {
            "image": image_probabilities,
            args.feature_model: tabular_probabilities,
            "fusion": fused_probabilities,
        }
        for branch_name, probabilities in branch_probabilities.items():
            _save_branch_charts(
                validation_labels,
                probabilities,
                fold_output / branch_name,
                title=f"Fold {fold} {branch_name}",
            )
            pooled[branch_name]["labels"].append(validation_labels.copy())
            pooled[branch_name]["probabilities"].append(probabilities.copy())

        print(f"[Fold {fold}] checkpoint: {checkpoint_path}", flush=True)
        for metric_name, value in fold_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}", flush=True)

        image_model.to("cpu")
        del (
            image_model,
            datamodule,
            preprocessor,
            tabular_model,
            tabular_feature_selector,
            tabular_features,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate_output = output_root / "aggregate"
    for branch_name, branch_data in pooled.items():
        branch_labels = np.concatenate(branch_data["labels"])
        branch_probabilities = np.concatenate(branch_data["probabilities"])
        _save_branch_charts(
            branch_labels,
            branch_probabilities,
            aggregate_output / branch_name,
            title=f"Pooled {branch_name}",
        )

    metrics_path = output_root / "validation_metrics.csv"
    pd.DataFrame(metrics_per_fold).to_csv(metrics_path, index=False)
    config_path = output_root / "validation_config.json"
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "experiment_name": experiment_name,
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    numeric_columns = pd.DataFrame(metrics_per_fold).select_dtypes(include="number")
    summary_rows = []
    print("\n========== Configurable fusion validation summary ==========", flush=True)
    for metric_name in numeric_columns.columns:
        if metric_name == "fold":
            continue
        values = numeric_columns[metric_name].dropna().to_numpy()
        if len(values):
            summary_rows.append(
                {
                    "metric": metric_name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
            )
            print(
                f"{metric_name}: mean={values.mean():.4f}, std={values.std():.4f}",
                flush=True,
            )
    summary_path = output_root / "validation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Metrics: {metrics_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Configuration: {config_path}", flush=True)
    return metrics_per_fold


def main():
    run_validation(parse_args())


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

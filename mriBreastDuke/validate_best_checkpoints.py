from pathlib import Path
import argparse
from numbers import Number
import math
import re
import os
from typing import List

from monai.networks.nets import DenseNet121
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pytorch_lightning as pl
import numpy as np
import torch
import matplotlib.pyplot as plt

from mriBreastDuke.constants import CHECKPOINTS, NIFTI_PATH, SEED
from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.dataLoaders import (
    NiftiDataModule,
    get_oncotype_score_for_series_as_studyId_and_label_df,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load best checkpoints from each CV fold and run validation."
    )
    parser.add_argument("--model", type=int, default=0, help="0=FCN, 1=DenseNet")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sensitivity_lambda", type=float, default=None)
    parser.add_argument("--positive_boost", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--top_k_checkpoints", type=int, default=3)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--is_binary_classification", type=bool, default=False)
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=CHECKPOINTS,
        help="Root directory that contains model fold checkpoints.",
    )
    parser.add_argument(
        "--charts_dir",
        type=str,
        default="validation_charts",
        help="Directory where fold confusion matrix and ROC charts will be saved.",
    )
    return parser.parse_args()


BEST_CKPT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-val_sensitivity=(?P<sensitivity>[0-9]*\.?[0-9]+)"
    r"(?:-val_auc_roc=(?P<auc>[0-9]*\.?[0-9]+))?\.ckpt$"
)


def get_top_checkpoints(checkpoint_dir: Path, top_k: int = 3) -> List[Path]:
    candidates = list(checkpoint_dir.glob("best-*.ckpt"))

    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt file found in: {checkpoint_dir}")

    parsed = []

    for path in candidates:
        match = BEST_CKPT_PATTERN.match(path.name)

        if match:
            sensitivity = float(match.group("sensitivity"))
            auc_value = match.group("auc")
            auc_value = float(auc_value) if auc_value is not None else -1.0
            epoch = int(match.group("epoch"))

            # Higher sensitivity is primary, AUC secondary, epoch tertiary.
            parsed.append((sensitivity, auc_value, epoch, path))
        else:
            # Fallback for unexpected names.
            parsed.append((-1.0, -1.0, -1, path))

    parsed.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    return [item[3] for item in parsed[:top_k]]


def resolve_fold_checkpoint_dir(checkpoint_root: str, model_name: str, fold: int) -> Path:
    root = Path(checkpoint_root)

    candidates = [
        root / model_name / f"fold_{fold}" / "checkpoints",
        root / f"fold_{fold}" / "checkpoints",
        root / f"fold_{fold}",
    ]

    if root.name == "checkpoints" and root.parent.name.startswith("fold_"):
        model_dir = root.parent.parent
        candidates.append(model_dir / f"fold_{fold}" / "checkpoints")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def configure_checkpoint_loading():
    # PyTorch 2.6 switched torch.load(weights_only=True) by default.
    # Lightning checkpoint files from previous runs may require full unpickling.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    safe_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ]
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_globals)


def save_confusion_matrix_chart(y_true: List[int], y_pred: List[int], output_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    threshold = cm.max() * 0.6 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_chart(y_true: List[int], y_prob: np.ndarray, num_classes: int, output_path: Path):
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))

    # -------------------------
    # Binary classification
    # -------------------------
    if num_classes == 2:
        # y_prob should usually be [N, 2] from softmax.
        # Use probability of positive class 1.
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            positive_probs = y_prob[:, 1]
        elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
            positive_probs = y_prob[:, 0]
        else:
            positive_probs = y_prob.reshape(-1)

        # ROC needs both classes present in y_true.
        if np.unique(y_true).size < 2:
            print(
                f"[ROC] Skipping binary ROC for {output_path}: only one class present in y_true.",
                flush=True,
            )
            plt.close(fig)
            return

        fpr, tpr, _ = roc_curve(y_true, positive_probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"class_1 positive ROC (AUC={roc_auc:.3f})")

    # -------------------------
    # Multiclass classification
    # -------------------------
    else:
        y_true_onehot = label_binarize(y_true, classes=list(range(num_classes)))

        for c in range(num_classes):
            if np.unique(y_true_onehot[:, c]).size < 2:
                print(
                    f"[ROC] Skipping class {c}: only one target value present.",
                    flush=True,
                )
                continue

            fpr, tpr, _ = roc_curve(y_true_onehot[:, c], y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"class_{c} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve" if num_classes == 2 else "ROC Curves (One-vs-Rest)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_saved_checkpoint_validation(
    df,
    model_name,
    make_model,
    num_folds,
    batch_size,
    num_workers,
    checkpoint_root,
    charts_dir,
    num_classes,
    top_k_checkpoints=3,
):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    y = df["label"].values
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n========== Fold {fold}/{num_folds} ==========")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        class_weights = _compute_balanced_class_weights(
            train_df["label"].values,
            num_classes=num_classes,
        )

        datamodule = NiftiDataModule(
            train_df=val_df,
            val_df=val_df,
            target_size=(256, 256, 64),
            image_root=NIFTI_PATH,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        checkpoint_dir = resolve_fold_checkpoint_dir(checkpoint_root, model_name, fold)
        checkpoint_paths = get_top_checkpoints(
            checkpoint_dir,
            top_k=top_k_checkpoints,
        )

        for checkpoint_rank, checkpoint_path in enumerate(checkpoint_paths, start=1):
            print(
                f"[Fold {fold}] Validating checkpoint rank {checkpoint_rank}: {checkpoint_path}",
                flush=True,
            )

            trainer = pl.Trainer(logger=False, enable_progress_bar=True)

            metrics = trainer.validate(
                model=make_model(class_weights=class_weights),
                datamodule=datamodule,
                ckpt_path=str(checkpoint_path),
                verbose=False,
                weights_only=False,
            )[0]

            model = trainer.lightning_module.eval()

            y_true = []
            y_pred = []
            y_prob = []
            device = model.device

            with torch.no_grad():
                for x, labels in datamodule.val_dataloader():
                    x = x.to(device)
                    logits = model(x)

                    if num_classes == 2 and (logits.ndim == 1 or logits.shape[1] == 1):
                        positive_probs = torch.sigmoid(logits.view(-1))
                        negative_probs = 1.0 - positive_probs

                        probs = torch.stack(
                            [negative_probs, positive_probs],
                            dim=1,
                        ).cpu().numpy()

                        preds = (positive_probs >= 0.5).long().cpu().numpy().tolist()

                    else:
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1).tolist()

                    y_prob.extend(probs.tolist())
                    y_pred.extend(preds)
                    y_true.extend(labels.cpu().numpy().tolist())

            fold_chart_dir = (
                    Path(charts_dir)
                    / model_name
                    / f"fold_{fold}"
                    / f"checkpoint_rank_{checkpoint_rank}"
            )
            fold_chart_dir.mkdir(parents=True, exist_ok=True)

            cm_path = fold_chart_dir / "confusion_matrix.png"
            roc_path = fold_chart_dir / "roc_curve.png"

            save_confusion_matrix_chart(y_true, y_pred, cm_path)
            save_roc_curve_chart(y_true, np.array(y_prob), num_classes, roc_path)

            serialized_metrics = {
                k: float(v) if isinstance(v, Number) else v
                for k, v in metrics.items()
            }

            serialized_metrics["fold"] = fold
            serialized_metrics["checkpoint_rank"] = checkpoint_rank
            serialized_metrics["checkpoint_path"] = str(checkpoint_path)
            serialized_metrics["confusion_matrix_path"] = str(cm_path)
            serialized_metrics["roc_curve_path"] = str(roc_path)

            fold_metrics.append(serialized_metrics)

            print(f"Checkpoint rank {checkpoint_rank}: {checkpoint_path}")
            for k, v in serialized_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        y_true = []
        y_pred = []
        y_prob = []
        device = model.device

        with torch.no_grad():
            for x, labels in datamodule.val_dataloader():
                x = x.to(device)
                logits = model(x)

                if num_classes == 2 and (logits.ndim == 1 or logits.shape[1] == 1):
                    # True binary model: one output logit.
                    positive_probs = torch.sigmoid(logits.view(-1))
                    negative_probs = 1.0 - positive_probs

                    probs = torch.stack([negative_probs, positive_probs], dim=1).cpu().numpy()
                    preds = (positive_probs >= 0.5).long().cpu().numpy().tolist()

                else:
                    # Multiclass model, or binary model with two logits.
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1).tolist()

                y_prob.extend(probs.tolist())
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy().tolist())

        fold_chart_dir = Path(charts_dir) / model_name / f"fold_{fold}"
        fold_chart_dir.mkdir(parents=True, exist_ok=True)

        cm_path = fold_chart_dir / "confusion_matrix.png"
        roc_path = fold_chart_dir / "roc_curve.png"
        save_confusion_matrix_chart(y_true, y_pred, cm_path)
        save_roc_curve_chart(y_true, np.array(y_prob), num_classes, roc_path)

        serialized_metrics = {
            k: float(v) if isinstance(v, Number) else v
            for k, v in metrics.items()
        }
        serialized_metrics["checkpoint_path"] = str(checkpoint_path)
        serialized_metrics["confusion_matrix_path"] = str(cm_path)
        serialized_metrics["roc_curve_path"] = str(roc_path)
        fold_metrics.append(serialized_metrics)

        print(f"Checkpoint: {checkpoint_path}")
        for k, v in serialized_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return fold_metrics

def _compute_balanced_class_weights(labels, num_classes):
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=num_classes)

    total = class_counts.sum().float()
    weights = total / (num_classes * class_counts.float().clamp_min(1.0))

    return weights

def print_summary(metrics_per_fold):
    print("\n========== Validation Summary ==========")
    keys = sorted({k for fold_metrics in metrics_per_fold for k in fold_metrics.keys()})

    for k in keys:
        values = [m[k] for m in metrics_per_fold if k in m]
        numeric_values = [
            v for v in values
            if isinstance(v, Number) and not (isinstance(v, float) and math.isnan(v))
        ]

        if numeric_values:
            mean = sum(numeric_values) / len(numeric_values)
            std = (sum((v - mean) ** 2 for v in numeric_values) / len(numeric_values)) ** 0.5
            print(f"{k}: mean={mean:.4f}, std={std:.4f}")

def build_model_name(base_name, args):
    """
    Builds the same model_name used during training.

    If hyperparameter args are not provided, returns base_name.
    """
    if args.lr is None:
        return base_name

    parts = [f"{base_name}_lr_{args.lr:.0e}"]

    if args.sensitivity_lambda is not None:
        parts.append(f"sens_{args.sensitivity_lambda}")

    if args.positive_boost is not None:
        parts.append(f"posboost_{args.positive_boost}")

    if args.weight_decay is not None:
        parts.append(f"wd_{args.weight_decay:.0e}")

    parts.append(f"bs_{args.batch_size}")

    return "_".join(parts)


def main():
    configure_checkpoint_loading()
    args = parse_args()
    df = get_oncotype_score_for_series_as_studyId_and_label_df(args.is_binary_classification)
    num_classes = len(set(df.label))

    models = [
        (
            "FCN",
            lambda class_weights=None: NiftiClassifier(
                Simple3DFCN(num_classes=num_classes),
                num_classes,
                class_weights=class_weights,
            ),
        ),
        (
            "DenseNet",
            lambda class_weights=None: NiftiClassifier(
                DenseNet121(
                    spatial_dims=3,
                    in_channels=5,
                    out_channels=num_classes,
                ),
                num_classes,
                class_weights=class_weights,
            ),
        ),
    ]

    base_model_name, make_model = models[args.model]
    model_name = build_model_name(base_model_name, args)

    print(f"[VALIDATION] Base model name: {base_model_name}", flush=True)
    print(f"[VALIDATION] Resolved checkpoint model name: {model_name}", flush=True)

    metrics_per_fold = run_saved_checkpoint_validation(
        df=df,
        model_name=model_name,
        make_model=make_model,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_root=args.checkpoint_root,
        charts_dir=args.charts_dir,
        num_classes=num_classes,
        top_k_checkpoints=args.top_k_checkpoints,
    )

    print_summary(metrics_per_fold)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

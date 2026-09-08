"""Leakage-safe inner calibration splits and binary threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def make_inner_calibration_split(
    labels,
    groups=None,
    requested_folds=5,
    random_state=42,
):
    """Split outer-training rows into fit and calibration indices.

    The first deterministic fold is held out for checkpoint selection and
    threshold tuning. When groups are supplied, no group can cross the split.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least 2.")

    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("Inner calibration requires at least two classes.")

    placeholder = np.zeros(labels.size, dtype=np.uint8)
    if groups is None:
        class_counts = np.array([np.sum(labels == value) for value in classes])
        effective_folds = min(int(requested_folds), int(class_counts.min()))
        if effective_folds < 2:
            raise ValueError(
                "Every class needs at least two rows for inner calibration."
            )
        splitter = StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=random_state,
        )
        fit_indices, calibration_indices = next(splitter.split(placeholder, labels))
    else:
        groups = np.asarray(groups)
        if groups.shape != labels.shape:
            raise ValueError("groups must align with labels.")
        if pd.isna(groups).any():
            raise ValueError("groups cannot contain missing values.")

        group_labels = pd.DataFrame({"group": groups, "label": labels})
        labels_per_group = group_labels.groupby("group", sort=False)["label"].nunique()
        if (labels_per_group > 1).any():
            raise ValueError("Each group must have exactly one label.")
        unique_groups = group_labels.drop_duplicates()
        groups_per_class = np.array(
            [
                unique_groups.loc[
                    unique_groups["label"] == value,
                    "group",
                ].nunique()
                for value in classes
            ]
        )
        effective_folds = min(int(requested_folds), int(groups_per_class.min()))
        if effective_folds < 2:
            raise ValueError(
                "Every class needs at least two patient groups for inner calibration."
            )
        splitter = StratifiedGroupKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=random_state,
        )
        fit_indices, calibration_indices = next(
            splitter.split(placeholder, labels, groups)
        )
        overlap = set(groups[fit_indices]).intersection(groups[calibration_indices])
        if overlap:
            raise RuntimeError("Patient-group leakage detected in inner calibration.")

    for name, indices in (
        ("fit", fit_indices),
        ("calibration", calibration_indices),
    ):
        present_classes = np.unique(labels[indices])
        if not np.array_equal(present_classes, classes):
            raise ValueError(
                f"The inner {name} split is missing a class: "
                f"received {present_classes.tolist()}, expected {classes.tolist()}."
            )

    return fit_indices, calibration_indices, effective_folds


def predictions_at_threshold(probabilities, threshold):
    """Convert positive-class probabilities to binary predictions."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be between 0 and 1.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return (probabilities >= threshold).astype(np.int64)


def select_balanced_accuracy_threshold(labels, probabilities):
    """Choose a binary threshold on calibration data only.

    Balanced accuracy is primary. Ties prefer higher sensitivity, then the
    threshold closest to 0.5, giving deterministic and clinically conservative
    behavior without consulting the outer validation fold.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be aligned 1-D arrays.")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Threshold tuning requires binary labels encoded as 0 and 1.")

    # Predictions change only when the threshold crosses an observed score.
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    records = []
    for threshold in candidates:
        predictions = predictions_at_threshold(probabilities, threshold)
        sensitivity = recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        )
        specificity = recall_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0,
        )
        records.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
            }
        )

    return max(
        records,
        key=lambda record: (
            record["balanced_accuracy"],
            record["sensitivity"],
            -abs(record["threshold"] - 0.5),
        ),
    )

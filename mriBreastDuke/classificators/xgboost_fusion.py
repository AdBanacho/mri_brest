"""Utilities for decision-level fusion of MRI and XGBoost probabilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)

from mriBreastDuke.threshold_tuning import predictions_at_threshold


def aligned_predict_proba(model, features, num_classes):
    """Return predict_proba output aligned to integer labels 0..num_classes-1."""
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    aligned = np.zeros((len(features), num_classes), dtype=np.float64)
    for source_index, class_label in enumerate(model.classes_):
        class_index = int(class_label)
        if class_index < 0 or class_index >= num_classes:
            raise ValueError(f"Unexpected XGBoost class label: {class_label}")
        aligned[:, class_index] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("XGBoost produced a row without class probability mass.")
    return aligned / row_sums


def fuse_probabilities(image_probabilities, xgboost_probabilities, alpha=0.5):
    """Combine class probabilities with ``alpha`` as the MRI branch weight."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    image_probabilities = np.asarray(image_probabilities, dtype=np.float64)
    xgboost_probabilities = np.asarray(xgboost_probabilities, dtype=np.float64)
    if image_probabilities.shape != xgboost_probabilities.shape:
        raise ValueError(
            "MRI and XGBoost probability arrays must have the same shape; "
            f"received {image_probabilities.shape} and {xgboost_probabilities.shape}."
        )
    if image_probabilities.ndim != 2 or image_probabilities.shape[1] < 2:
        raise ValueError("Probability arrays must have shape (samples, classes).")
    if not np.all(np.isfinite(image_probabilities)) or not np.all(
        np.isfinite(xgboost_probabilities)
    ):
        raise ValueError("Probability arrays must contain only finite values.")

    fused = alpha * image_probabilities + (1.0 - alpha) * xgboost_probabilities
    row_sums = fused.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Fused predictions contain a row without probability mass.")
    return fused / row_sums


def probability_predictions(probabilities, threshold=None):
    """Return argmax predictions or apply a tuned binary threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if threshold is None:
        return np.argmax(probabilities, axis=1)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("A decision threshold can only be used for binary probabilities.")
    return predictions_at_threshold(probabilities[:, 1], threshold)


def probability_metrics(labels, probabilities, prefix, threshold=None):
    """Calculate classification metrics for one probability matrix."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probability_predictions(probabilities, threshold=threshold)
    sensitivity_options = (
        {"average": "binary", "pos_label": 1}
        if probabilities.shape[1] == 2
        else {"average": "macro"}
    )
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(labels, predictions)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        f"{prefix}_sensitivity": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
                **sensitivity_options,
            )
        ),
    }
    if threshold is not None:
        metrics[f"{prefix}_decision_threshold"] = float(threshold)
    if probabilities.shape[1] == 2:
        metrics[f"{prefix}_specificity"] = float(
            recall_score(
                labels,
                predictions,
                average="binary",
                pos_label=0,
                zero_division=0,
            )
        )
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(
                labels,
                probabilities,
                average="macro",
                multi_class="ovr",
                labels=np.arange(probabilities.shape[1]),
            )
        metrics[f"{prefix}_auc_roc"] = float(auc)
    except ValueError:
        metrics[f"{prefix}_auc_roc"] = float("nan")
    return metrics


def save_fusion_predictions(
    validation_data,
    labels,
    image_probabilities,
    xgboost_probabilities,
    fused_probabilities,
    output_path,
    tabular_model_name="xgboost",
    decision_thresholds=None,
):
    """Save identifiers, branch probabilities, and fused predictions to CSV."""
    identifiers = {}
    for column in ("patientId", "studyId"):
        if column in validation_data.columns:
            identifiers[column] = validation_data[column].to_numpy()

    tabular_model_name = str(tabular_model_name).strip()
    if not tabular_model_name:
        raise ValueError("tabular_model_name cannot be empty.")

    output = pd.DataFrame({**identifiers, "label": labels})
    for class_index in range(fused_probabilities.shape[1]):
        output[f"image_probability_{class_index}"] = image_probabilities[:, class_index]
        output[f"{tabular_model_name}_probability_{class_index}"] = (
            xgboost_probabilities[:, class_index]
        )
        output[f"fusion_probability_{class_index}"] = fused_probabilities[:, class_index]
    decision_thresholds = decision_thresholds or {}
    output["image_prediction"] = probability_predictions(
        image_probabilities,
        threshold=decision_thresholds.get("image"),
    )
    output[f"{tabular_model_name}_prediction"] = probability_predictions(
        xgboost_probabilities,
        threshold=decision_thresholds.get(tabular_model_name),
    )
    output["fusion_prediction"] = probability_predictions(
        fused_probabilities,
        threshold=decision_thresholds.get("fusion"),
    )
    for branch_name, threshold in decision_thresholds.items():
        output[f"{branch_name}_decision_threshold"] = float(threshold)
    output.to_csv(output_path, index=False)

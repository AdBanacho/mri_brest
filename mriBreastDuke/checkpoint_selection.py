"""Shared ranking for current and legacy validation checkpoint filenames."""

import re


METRIC_VALUE_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
AUC_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    rf"val_auc_roc=(?P<auc>{METRIC_VALUE_PATTERN})-"
    rf"val_balanced_accuracy=(?P<balanced_accuracy>{METRIC_VALUE_PATTERN})-"
    rf"val_sensitivity=(?P<sensitivity>{METRIC_VALUE_PATTERN})"
    r"(?:-v\d+)?\.ckpt$"
)
BALANCED_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    rf"(?:val_balanced_accuracy=(?P<balanced_accuracy>{METRIC_VALUE_PATTERN})-)?"
    rf"val_sensitivity=(?P<sensitivity>{METRIC_VALUE_PATTERN})"
    rf"(?:-val_auc_roc=(?P<auc>{METRIC_VALUE_PATTERN}))?"
    r"(?:-v\d+)?\.ckpt$"
)


def checkpoint_score(path):
    """Return a sortable score, preferring current AUC checkpoints.

    Balanced-accuracy and sensitivity-era checkpoints remain supported. A
    current AUC-format checkpoint always ranks ahead of an older format so a
    stale checkpoint cannot override a newly trained model.
    """
    match = AUC_CHECKPOINT_PATTERN.match(path.name)
    if match is not None:
        return (
            1,
            2,
            float(match.group("auc")),
            float(match.group("balanced_accuracy")),
            float(match.group("sensitivity")),
            int(match.group("epoch")),
            path.stat().st_mtime,
        )

    match = BALANCED_CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return (
            0,
            0,
            float("-inf"),
            float("-inf"),
            float("-inf"),
            -1,
            path.stat().st_mtime,
        )

    sensitivity = float(match.group("sensitivity"))
    auc_value = match.group("auc")
    auc_value = float(auc_value) if auc_value is not None else float("-inf")
    epoch = int(match.group("epoch"))
    balanced_accuracy = match.group("balanced_accuracy")
    if balanced_accuracy is None:
        return (
            1,
            0,
            sensitivity,
            auc_value,
            float("-inf"),
            epoch,
            path.stat().st_mtime,
        )

    return (
        1,
        1,
        float(balanced_accuracy),
        sensitivity,
        auc_value,
        epoch,
        path.stat().st_mtime,
    )


def rank_checkpoints(paths):
    """Return checkpoint paths from best to worst according to their format."""
    return sorted(paths, key=checkpoint_score, reverse=True)

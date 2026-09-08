"""Shared ranking for current and legacy validation checkpoint filenames."""

import re


METRIC_VALUE_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BEST_CHECKPOINT_PATTERN = re.compile(
    r"best-epoch=(?P<epoch>\d+)-"
    rf"(?:val_balanced_accuracy=(?P<balanced_accuracy>{METRIC_VALUE_PATTERN})-)?"
    rf"val_sensitivity=(?P<sensitivity>{METRIC_VALUE_PATTERN})"
    rf"(?:-val_auc_roc=(?P<auc>{METRIC_VALUE_PATTERN}))?"
    r"(?:-v\d+)?\.ckpt$"
)


def checkpoint_score(path):
    """Return a sortable score, preferring balanced-accuracy checkpoints.

    Checkpoints created before balanced accuracy was logged remain supported.
    A current-format checkpoint always ranks ahead of a legacy checkpoint so
    stale sensitivity-selected files cannot override a newly trained model.
    """
    match = BEST_CHECKPOINT_PATTERN.match(path.name)
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

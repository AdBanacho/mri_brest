SUBTRACTION_NONE = "none"
SUBTRACTION_POST_MINUS_PRE = "post_minus_pre"
SUBTRACTION_CONSECUTIVE = "consecutive"

SUBTRACTION_MODES = (
    SUBTRACTION_NONE,
    SUBTRACTION_POST_MINUS_PRE,
    SUBTRACTION_CONSECUTIVE,
)


def validate_subtraction_mode(mode):
    if mode not in SUBTRACTION_MODES:
        raise ValueError(
            f"Unknown subtraction mode {mode!r}; expected one of {SUBTRACTION_MODES}."
        )
    return mode


def get_input_channels(mode, max_series=5):
    validate_subtraction_mode(mode)
    if mode == SUBTRACTION_NONE:
        return max_series
    if max_series < 2:
        raise ValueError("Subtraction requires at least two series.")
    return max_series - 1


def build_subtraction_pairs(series_ids, mode):
    """Return ordered ``(minuend, subtrahend)`` series pairs."""
    validate_subtraction_mode(mode)
    series_ids = list(series_ids)

    if mode == SUBTRACTION_NONE:
        return []
    if len(series_ids) < 2:
        raise ValueError(
            f"Subtraction mode {mode!r} requires at least two MRI series; "
            f"received {len(series_ids)}."
        )

    if mode == SUBTRACTION_POST_MINUS_PRE:
        pre_contrast = series_ids[0]
        return [(post_contrast, pre_contrast) for post_contrast in series_ids[1:]]

    return list(zip(series_ids[1:], series_ids[:-1]))

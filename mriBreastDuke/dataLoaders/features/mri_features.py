"""Handcrafted features for registered dynamic contrast-enhanced breast MRI.

The functions in this module operate on raw MRI intensities. Do not independently
z-score each DCE phase before calculating kinetic features because doing so removes
the between-phase enhancement signal.
"""

import os

import numpy as np
from scipy import ndimage


_GLCM_OFFSETS_3D = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
)


def _validate_mask(mask, expected_shape=None):
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"lesion_mask must be 3D; received shape {mask.shape}.")
    if expected_shape is not None and mask.shape != tuple(expected_shape):
        raise ValueError(
            "lesion_mask and MRI geometry differ: "
            f"mask={mask.shape}, MRI={tuple(expected_shape)}."
        )
    mask = mask.astype(bool)
    if not np.any(mask):
        raise ValueError("lesion_mask is empty.")
    return mask


def _validate_spacing(voxel_spacing):
    spacing = np.asarray(voxel_spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
        raise ValueError("voxel_spacing must contain three finite values in mm.")
    if np.any(spacing <= 0):
        raise ValueError("voxel_spacing values must be greater than zero.")
    return spacing


def _safe_ratio(numerator, denominator, epsilon=1e-8):
    if abs(denominator) <= epsilon:
        return float("nan")
    return float(numerator / denominator)


def extract_temporal_kinetic_features(
    volumes,
    lesion_mask,
    time_points=None,
    curve_threshold=0.10,
):
    """Extract ROI-level enhancement features from an ordered DCE series.

    Args:
        volumes: Array shaped ``(time, x, y, z)``. Index 0 must be pre-contrast
            and the remaining volumes must be registered post-contrast phases.
        lesion_mask: Binary 3D tumor/lesion mask in the same geometry.
        time_points: Optional acquisition times in minutes. If omitted, phase
            indices are used, so slope units are intensity per phase.
        curve_threshold: Relative late-change threshold used to distinguish
            persistent, plateau, and washout curves.

    Returns:
        A flat feature dictionary. Curve codes are 0=persistent, 1=plateau,
        and 2=washout.
    """
    volumes = np.asarray(volumes, dtype=np.float64)
    if volumes.ndim != 4 or volumes.shape[0] < 2:
        raise ValueError(
            "volumes must have shape (time, x, y, z) with at least one "
            "pre-contrast and one post-contrast phase."
        )
    mask = _validate_mask(lesion_mask, volumes.shape[1:])

    if time_points is None:
        times = np.arange(volumes.shape[0], dtype=np.float64)
    else:
        times = np.asarray(time_points, dtype=np.float64)
        if times.shape != (volumes.shape[0],):
            raise ValueError("time_points length must match the number of DCE phases.")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("time_points must be finite and strictly increasing.")

    finite_roi = mask & np.all(np.isfinite(volumes), axis=0)
    if not np.any(finite_roi):
        raise ValueError("No lesion voxels have finite values in every DCE phase.")

    signal = np.mean(volumes[:, finite_roi], axis=1)
    baseline = signal[0]
    baseline_scale = max(abs(baseline), 1e-8)
    enhancement_percent = 100.0 * (signal - baseline) / baseline_scale

    post_signal = signal[1:]
    peak_index = int(np.argmax(post_signal)) + 1
    early_signal = signal[1]
    late_signal = signal[-1]
    peak_signal = signal[peak_index]

    consecutive_slopes = np.diff(signal) / np.diff(times)
    post_peak_slopes = consecutive_slopes[peak_index:]
    maximum_washout_slope = (
        float(np.min(post_peak_slopes)) if post_peak_slopes.size else 0.0
    )

    early_amplitude = early_signal - baseline
    relative_late_change = _safe_ratio(late_signal - early_signal, abs(early_amplitude))
    if not np.isfinite(relative_late_change):
        curve_name = "plateau"
        curve_code = 1
    elif relative_late_change > curve_threshold:
        curve_name = "persistent"
        curve_code = 0
    elif relative_late_change < -curve_threshold:
        curve_name = "washout"
        curve_code = 2
    else:
        curve_name = "plateau"
        curve_code = 1

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return {
        "baseline_mean": float(baseline),
        "early_post_mean": float(early_signal),
        "late_post_mean": float(late_signal),
        "peak_mean": float(peak_signal),
        "early_enhancement_percent": float(enhancement_percent[1]),
        "late_enhancement_percent": float(enhancement_percent[-1]),
        "peak_enhancement_percent": float(enhancement_percent[peak_index]),
        "time_to_peak": float(times[peak_index] - times[0]),
        "maximum_wash_in_slope": float(np.max(consecutive_slopes[:peak_index])),
        "maximum_washout_slope": maximum_washout_slope,
        "signal_enhancement_ratio": _safe_ratio(
            early_signal - baseline,
            late_signal - baseline,
        ),
        "relative_late_change": relative_late_change,
        "enhancement_auc": float(trapezoid(enhancement_percent, times)),
        "kinetic_curve_type": curve_name,
        "kinetic_curve_type_code": curve_code,
    }


def _surface_area(mask, spacing):
    surface_area = 0.0
    for axis in range(3):
        face_area = float(np.prod(np.delete(spacing, axis)))
        adjacent_a = [slice(None)] * 3
        adjacent_b = [slice(None)] * 3
        adjacent_a[axis] = slice(1, None)
        adjacent_b[axis] = slice(None, -1)
        transitions = np.count_nonzero(
            mask[tuple(adjacent_a)] != mask[tuple(adjacent_b)]
        )

        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        boundary_faces = np.count_nonzero(mask[tuple(first)]) + np.count_nonzero(
            mask[tuple(last)]
        )
        surface_area += (transitions + boundary_faces) * face_area
    return float(surface_area)


def extract_morphological_features(lesion_mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """Extract physical 3D shape features from a binary lesion mask."""
    mask = _validate_mask(lesion_mask)
    spacing = _validate_spacing(voxel_spacing)

    voxel_count = int(np.count_nonzero(mask))
    voxel_volume = float(np.prod(spacing))
    volume = voxel_count * voxel_volume
    surface_area = _surface_area(mask, spacing)
    equivalent_diameter = float(2.0 * ((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0))
    sphericity = float(
        (np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / surface_area
    )

    coordinates = np.argwhere(mask).astype(np.float64) * spacing
    extents = (np.ptp(np.argwhere(mask), axis=0) + 1) * spacing
    bbox_diagonal = float(np.linalg.norm(extents))

    if voxel_count > 1:
        covariance = np.cov(coordinates, rowvar=False, bias=True)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        principal_stds = np.sqrt(np.clip(eigenvalues, 0.0, None))
    else:
        principal_stds = np.zeros(3, dtype=np.float64)

    structure = ndimage.generate_binary_structure(3, 1)
    components, component_count = ndimage.label(mask, structure=structure)
    component_sizes = np.bincount(components.ravel())[1:]
    largest_component_fraction = float(component_sizes.max() / voxel_count)

    return {
        "voxel_count": voxel_count,
        "volume_mm3": float(volume),
        "surface_area_mm2": surface_area,
        "surface_to_volume_ratio_per_mm": float(surface_area / volume),
        "sphericity": sphericity,
        "equivalent_sphere_diameter_mm": equivalent_diameter,
        "bounding_box_x_mm": float(extents[0]),
        "bounding_box_y_mm": float(extents[1]),
        "bounding_box_z_mm": float(extents[2]),
        "bounding_box_diagonal_mm": bbox_diagonal,
        "principal_axis_std_major_mm": float(principal_stds[0]),
        "principal_axis_std_intermediate_mm": float(principal_stds[1]),
        "principal_axis_std_minor_mm": float(principal_stds[2]),
        "elongation": _safe_ratio(principal_stds[1], principal_stds[0]),
        "flatness": _safe_ratio(principal_stds[2], principal_stds[0]),
        "connected_component_count": int(component_count),
        "largest_component_fraction": largest_component_fraction,
    }


def _quantize_roi(image, mask, levels, intensity_range):
    values = image[mask]
    if intensity_range is None:
        lower, upper = np.percentile(values, (1.0, 99.0))
    else:
        if len(intensity_range) != 2:
            raise ValueError("intensity_range must be a (minimum, maximum) pair.")
        lower, upper = map(float, intensity_range)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("intensity_range must contain finite values with maximum >= minimum.")

    quantized = np.zeros(image.shape, dtype=np.int32)
    if upper > lower:
        scaled = (np.clip(image[mask], lower, upper) - lower) / (upper - lower)
        quantized[mask] = np.minimum((scaled * levels).astype(np.int32), levels - 1)
    return quantized, float(lower), float(upper)


def _offset_slices(shape, offset):
    source = []
    neighbor = []
    for size, delta in zip(shape, offset):
        if delta >= 0:
            source.append(slice(0, size - delta))
            neighbor.append(slice(delta, size))
        else:
            source.append(slice(-delta, size))
            neighbor.append(slice(0, size + delta))
    return tuple(source), tuple(neighbor)


def _merged_glcm(quantized, mask, levels, distance):
    matrix = np.zeros((levels, levels), dtype=np.float64)
    valid_pairs = 0
    for base_offset in _GLCM_OFFSETS_3D:
        offset = tuple(distance * value for value in base_offset)
        if any(abs(delta) >= size for size, delta in zip(mask.shape, offset)):
            continue
        source_slice, neighbor_slice = _offset_slices(mask.shape, offset)
        pair_mask = mask[source_slice] & mask[neighbor_slice]
        if not np.any(pair_mask):
            continue
        source_values = quantized[source_slice][pair_mask]
        neighbor_values = quantized[neighbor_slice][pair_mask]
        np.add.at(matrix, (source_values, neighbor_values), 1)
        np.add.at(matrix, (neighbor_values, source_values), 1)
        valid_pairs += 2 * source_values.size

    if valid_pairs == 0:
        raise ValueError(
            "The lesion mask has no valid voxel pairs for the requested GLCM distance."
        )
    return matrix / matrix.sum(), valid_pairs


def extract_heterogeneity_features(
    image,
    lesion_mask,
    levels=32,
    distance=1,
    intensity_range=None,
):
    """Extract first-order and merged 3D GLCM features inside the lesion.

    Pass a cohort-wide ``intensity_range`` for directly comparable GLCM values.
    If it is omitted, the 1st and 99th percentiles of each lesion are used,
    which is robust for exploration but should be fixed within training folds.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image must be 3D; received shape {image.shape}.")
    mask = _validate_mask(lesion_mask, image.shape) & np.isfinite(image)
    if not np.any(mask):
        raise ValueError("No finite image values are present inside lesion_mask.")
    if not isinstance(levels, (int, np.integer)) or levels < 2:
        raise ValueError("levels must be an integer greater than or equal to 2.")
    if not isinstance(distance, (int, np.integer)) or distance < 1:
        raise ValueError("distance must be a positive integer.")

    values = image[mask]
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    if std > 1e-12:
        skewness = float(np.mean(centered ** 3) / (std ** 3))
        excess_kurtosis = float(np.mean(centered ** 4) / (std ** 4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    quantized, lower, upper = _quantize_roi(image, mask, levels, intensity_range)
    histogram = np.bincount(quantized[mask], minlength=levels).astype(np.float64)
    probabilities = histogram / histogram.sum()
    nonzero_probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero_probabilities * np.log2(nonzero_probabilities)))
    uniformity = float(np.sum(probabilities ** 2))

    glcm, valid_pair_count = _merged_glcm(quantized, mask, levels, distance)
    i, j = np.indices(glcm.shape, dtype=np.float64)
    difference = i - j
    glcm_mean_i = float(np.sum(i * glcm))
    glcm_mean_j = float(np.sum(j * glcm))
    glcm_std_i = float(np.sqrt(np.sum(((i - glcm_mean_i) ** 2) * glcm)))
    glcm_std_j = float(np.sqrt(np.sum(((j - glcm_mean_j) ** 2) * glcm)))
    correlation_denominator = glcm_std_i * glcm_std_j
    glcm_correlation = (
        float(np.sum((i - glcm_mean_i) * (j - glcm_mean_j) * glcm) / correlation_denominator)
        if correlation_denominator > 1e-12
        else 0.0
    )

    percentiles = np.percentile(values, (10, 25, 50, 75, 90))
    return {
        "mean": mean,
        "std": std,
        "coefficient_of_variation": _safe_ratio(std, abs(mean)),
        "minimum": float(np.min(values)),
        "percentile_10": float(percentiles[0]),
        "percentile_25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "percentile_75": float(percentiles[3]),
        "percentile_90": float(percentiles[4]),
        "maximum": float(np.max(values)),
        "interquartile_range": float(percentiles[3] - percentiles[1]),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "entropy": entropy,
        "uniformity": uniformity,
        "glcm_contrast": float(np.sum((difference ** 2) * glcm)),
        "glcm_dissimilarity": float(np.sum(np.abs(difference) * glcm)),
        "glcm_homogeneity": float(np.sum(glcm / (1.0 + difference ** 2))),
        "glcm_asm": float(np.sum(glcm ** 2)),
        "glcm_energy": float(np.sqrt(np.sum(glcm ** 2))),
        "glcm_correlation": glcm_correlation,
        "glcm_valid_pair_count": int(valid_pair_count),
        "quantization_minimum": lower,
        "quantization_maximum": upper,
    }


def _prefix_features(features, prefix):
    return {f"{prefix}_{name}": value for name, value in features.items()}


def extract_mri_features(
    volumes,
    lesion_mask,
    voxel_spacing=(1.0, 1.0, 1.0),
    time_points=None,
    heterogeneity_source="first_post_subtraction",
    levels=32,
    distance=1,
    intensity_range=None,
    curve_threshold=0.10,
):
    """Extract kinetic, morphology, and heterogeneity features together."""
    volumes = np.asarray(volumes, dtype=np.float64)
    if volumes.ndim != 4 or volumes.shape[0] < 2:
        raise ValueError("volumes must contain pre- and post-contrast 3D phases.")

    source_images = {
        "pre_contrast": volumes[0],
        "first_post": volumes[1],
        "first_post_subtraction": volumes[1] - volumes[0],
        "late_post": volumes[-1],
        "late_post_subtraction": volumes[-1] - volumes[0],
    }
    if heterogeneity_source not in source_images:
        choices = ", ".join(sorted(source_images))
        raise ValueError(f"Unknown heterogeneity_source '{heterogeneity_source}'. Choose: {choices}.")

    kinetic = extract_temporal_kinetic_features(
        volumes,
        lesion_mask,
        time_points=time_points,
        curve_threshold=curve_threshold,
    )
    morphology = extract_morphological_features(lesion_mask, voxel_spacing)
    heterogeneity = extract_heterogeneity_features(
        source_images[heterogeneity_source],
        lesion_mask,
        levels=levels,
        distance=distance,
        intensity_range=intensity_range,
    )

    return {
        **_prefix_features(kinetic, "kinetic"),
        **_prefix_features(morphology, "morphology"),
        **_prefix_features(heterogeneity, "heterogeneity"),
        "heterogeneity_source": heterogeneity_source,
    }


def extract_mri_features_from_nifti(
    series_paths,
    lesion_mask_path,
    time_points=None,
    **feature_options,
):
    """Load an ordered NIfTI DCE series and extract all feature families."""
    import nibabel as nib

    series_paths = [os.fspath(path) for path in series_paths]
    if len(series_paths) < 2:
        raise ValueError("series_paths must contain pre-contrast and post-contrast NIfTI files.")

    images = [nib.load(path) for path in series_paths]
    reference_shape = images[0].shape
    reference_affine = np.asarray(images[0].affine)
    for path, image in zip(series_paths[1:], images[1:]):
        if image.shape != reference_shape or not np.allclose(
            image.affine,
            reference_affine,
            rtol=1e-4,
            atol=1e-3,
        ):
            raise ValueError(
                f"DCE phase '{path}' is not registered to the pre-contrast geometry."
            )

    mask_image = nib.load(os.fspath(lesion_mask_path))
    if mask_image.shape != reference_shape or not np.allclose(
        mask_image.affine,
        reference_affine,
        rtol=1e-4,
        atol=1e-3,
    ):
        raise ValueError("The lesion mask is not registered to the DCE series geometry.")

    volumes = np.stack(
        [np.asarray(image.get_fdata(dtype=np.float32)) for image in images],
        axis=0,
    )
    lesion_mask = np.asarray(mask_image.get_fdata()) > 0
    voxel_spacing = images[0].header.get_zooms()[:3]
    return extract_mri_features(
        volumes,
        lesion_mask,
        voxel_spacing=voxel_spacing,
        time_points=time_points,
        **feature_options,
    )


def extract_feature_table(
    studies,
    image_root,
    mask_path_column="lesion_mask_path",
    series_column="series_ids",
    image_suffix=".nii.gz",
    time_points_column=None,
    **feature_options,
):
    """Extract one flat feature row per study from a pipeline DataFrame.

    ``studies`` must contain an ordered series-ID list and a lesion-mask path.
    Identifier and label columns are preserved in the returned DataFrame.
    """
    import pandas as pd

    required = {series_column, mask_path_column}
    missing = required.difference(studies.columns)
    if missing:
        raise ValueError(f"studies is missing required columns: {sorted(missing)}")

    rows = []
    for row_index, row in studies.iterrows():
        series_ids = list(row[series_column])
        series_paths = [
            os.path.join(os.fspath(image_root), f"{series_id}{image_suffix}")
            for series_id in series_ids
        ]
        time_points = row[time_points_column] if time_points_column else None
        try:
            features = extract_mri_features_from_nifti(
                series_paths,
                row[mask_path_column],
                time_points=time_points,
                **feature_options,
            )
        except Exception as error:
            study_id = row.get("studyId", row_index)
            raise RuntimeError(f"Feature extraction failed for study '{study_id}'.") from error

        identifiers = row.drop(labels=[series_column, mask_path_column]).to_dict()
        rows.append({**identifiers, **features})

    return pd.DataFrame(rows)

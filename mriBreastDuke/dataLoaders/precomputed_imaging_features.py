"""Load the Duke precomputed imaging-feature workbook by feature family."""

from pathlib import Path

import pandas as pd


IMAGING_FEATURE_GROUPS = ("kinetic", "morphology", "heterogeneity")

_MORPHOLOGY_COLUMNS = {
    "TumorMajorAxisLength_mm",
    "BEVR_Tumor",
    "BEDR1_Tumor",
    "BEDR2_Tumor",
    "MF_Tumor",
    "ASD_Tumor",
    "Volume_cu_mm_Tumor",
    "Median_solidity_Tumor",
    "Median_Elongation_Tumor",
    "Median_Euler_No_Tumor",
    "BreastVol",
    "tissueVol_T1",
    "tissueVol_PostCon",
    "breastDensity_T1",
    "breastDensity_PostCon",
}

_HETEROGENEITY_MARKERS = (
    "autocorrelation",
    "contrast",
    "correlation1",
    "correlation2",
    "cluster_prominence",
    "cluster_shade",
    "dissimilarity",
    "energy",
    "entropy",
    "homogeneity1",
    "homogeneity2",
    "max_probability",
    "sum_of_squares",
    "sum_avg",
    "sum_average",
    "sum_variance",
    "sum_entropy",
    "diff_entropy",
    "difference_entropy",
    "inf_mea_of_corr",
    "information_measure_correlation",
    "inv_diff",
    "inverse_difference",
    "globalmoransi",
    "enhancementcluster",
    "dft_",
    "dhog_",
    "dlbp_",
    "variance_of_uptake",
    "change_in_variance_of_uptake",
    "margin_gradient",
    "variance_of_margin_gradient",
    "variance_of_rgh",
    "_map_mean",
    "_map_std_dev",
    "_map_skewness",
    "_map_kurtosis",
)

_KINETIC_MARKERS = (
    "f1_dt_",
    "max_enhancement",
    "time_to_peak",
    "uptake_rate",
    "washout_rate",
    "ratio_tissue_vol_enhancing",
    "maximum_variance_of_enhancement",
    "peak_location_of_enhancement",
    "enhancement_variance_",
    "grouping_based_",
    "ser_total",
    "ser_partial",
    "ser_washout",
    "peak_ser",
    "peak_pe",
)


def classify_imaging_feature(column):
    """Map one Duke workbook column to a non-overlapping feature family."""
    column = str(column)
    lowered = column.lower()
    if column in _MORPHOLOGY_COLUMNS:
        return "morphology"
    if any(marker in lowered for marker in _HETEROGENEITY_MARKERS):
        return "heterogeneity"
    if any(marker in lowered for marker in _KINETIC_MARKERS):
        return "kinetic"
    raise ValueError(
        f"Imaging feature '{column}' is not assigned to morphology, kinetic, "
        "or heterogeneity. Update the feature-family mapping explicitly."
    )


def classify_imaging_feature_columns(columns):
    """Return source column names grouped into deterministic feature families."""
    grouped = {group: [] for group in IMAGING_FEATURE_GROUPS}
    for column in columns:
        grouped[classify_imaging_feature(column)].append(column)
    return grouped


def _read_feature_table(path):
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Imaging Features")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        "Precomputed imaging features must be supplied as .xlsx, .xls, or .csv."
    )


def load_precomputed_imaging_features(
    path,
    selected_groups,
    source_patient_id_column="Patient ID",
    output_patient_id_column="patientId",
):
    """Load and prefix selected feature families from Imaging_Features.xlsx."""
    selected_groups = tuple(dict.fromkeys(selected_groups))
    unknown = set(selected_groups).difference(IMAGING_FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown imaging feature groups: {sorted(unknown)}")
    if not selected_groups:
        raise ValueError("At least one imaging feature group must be selected.")

    table = _read_feature_table(path)
    if source_patient_id_column not in table.columns:
        raise ValueError(
            f"Imaging feature table is missing '{source_patient_id_column}'."
        )
    if table[source_patient_id_column].isna().any():
        raise ValueError("Imaging feature table contains a missing patient ID.")
    if table[source_patient_id_column].duplicated().any():
        raise ValueError("Imaging feature table contains duplicate patient IDs.")

    source_feature_columns = [
        column for column in table.columns if column != source_patient_id_column
    ]
    non_numeric = [
        column
        for column in source_feature_columns
        if not pd.api.types.is_numeric_dtype(table[column])
    ]
    if non_numeric:
        raise ValueError(
            "Imaging feature columns must be numeric; received non-numeric "
            f"columns: {non_numeric[:10]}"
        )

    grouped = classify_imaging_feature_columns(source_feature_columns)
    selected_source_columns = [
        column
        for group in selected_groups
        for column in grouped[group]
    ]
    renamed_columns = {
        column: f"{classify_imaging_feature(column)}_{column}"
        for column in selected_source_columns
    }

    selected = table[
        [source_patient_id_column, *selected_source_columns]
    ].copy()
    selected.rename(
        columns={
            source_patient_id_column: output_patient_id_column,
            **renamed_columns,
        },
        inplace=True,
    )
    selected[output_patient_id_column] = selected[
        output_patient_id_column
    ].astype(str)
    feature_columns = [renamed_columns[column] for column in selected_source_columns]

    print(
        "[IMAGING FEATURES] "
        + ", ".join(
            f"{group}={len(grouped[group])}"
            for group in selected_groups
        )
        + f"; selected={len(feature_columns)} rows={len(selected)}",
        flush=True,
    )
    return selected, feature_columns


def merge_precomputed_imaging_features(
    studies,
    path,
    selected_groups,
    patient_id_column="patientId",
    source_patient_id_column="Patient ID",
    require_complete_match=True,
):
    """Merge selected workbook features into a study-level modeling table."""
    if patient_id_column not in studies.columns:
        raise ValueError(f"Studies table is missing '{patient_id_column}'.")

    imaging, feature_columns = load_precomputed_imaging_features(
        path,
        selected_groups=selected_groups,
        source_patient_id_column=source_patient_id_column,
        output_patient_id_column=patient_id_column,
    )
    prepared_studies = studies.copy()
    prepared_studies[patient_id_column] = prepared_studies[
        patient_id_column
    ].astype(str)
    merged = prepared_studies.merge(
        imaging,
        on=patient_id_column,
        how="left",
        validate="many_to_one",
    )
    matched = int(merged[feature_columns].notna().any(axis=1).sum())
    if matched == 0:
        raise ValueError(
            "Imaging_Features did not match any studies by patient ID."
        )
    if require_complete_match and matched != len(merged):
        unmatched = merged.loc[
            ~merged[feature_columns].notna().any(axis=1),
            patient_id_column,
        ].astype(str)
        raise ValueError(
            f"Imaging features are missing for {len(unmatched)} studies, "
            f"including: {', '.join(unmatched.head(5))}"
        )

    print(
        f"[IMAGING FEATURES] matched={matched}/{len(merged)} studies",
        flush=True,
    )
    return merged, feature_columns

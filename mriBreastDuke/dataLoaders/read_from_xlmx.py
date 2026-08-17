import os

import pandas as pd

from mriBreastDuke.constants import (
    FEATURES_PATH,
    IMAGES_METADATA,
    MAX_SERIES_PER_STUDY,
    TARGETS_FILE_NAME,
)


# The clinical workbook contains three non-data rows: feature group, feature
# name, and coding definition. Reading the relevant Excel columns explicitly
# makes the loader independent of pandas' multi-index header flattening.
_CLINICAL_EXCEL_COLUMNS = "A,B,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AW,AX,AY,AZ,BA"
_CLINICAL_SOURCE_COLUMNS = (
    "patientId",
    "days_to_mri",
    "date_of_birth_days",
    "menopause",
    "race_ethnicity",
    "metastatic_at_presentation",
    "er_status",
    "pr_status",
    "her2_status",
    "molecular_subtype",
    "oncotype_score",
    "clinical_t_stage",
    "clinical_n_stage",
    "clinical_m_stage",
    "tubule_grade",
    "nuclear_grade",
    "mitotic_grade",
    "nottingham_grade",
    "histologic_type",
    "tumor_laterality",
    "multicentric_multifocal",
    "contralateral_breast_involvement",
    "suspicious_lymph_nodes",
    "skin_nipple_involvement",
    "pectoral_chest_involvement",
)

# Pretreatment variables suitable for multimodal models. Although the workbook
# represents most of these fields with numbers, they are categorical codes and
# should be encoded inside each training fold rather than treated as continuous.
CLINICAL_PREDICTOR_COLUMNS = (
    "age_at_diagnosis_years",
    "menopause",
    "metastatic_at_presentation",
    "er_status",
    "pr_status",
    "her2_status",
    "molecular_subtype",
    "clinical_t_stage",
    "clinical_n_stage",
    "clinical_m_stage",
    "tubule_grade",
    "nuclear_grade",
    "mitotic_grade",
    "nottingham_grade",
    "histologic_type",
    "multicentric_multifocal",
    "contralateral_breast_involvement",
    "suspicious_lymph_nodes",
    "skin_nipple_involvement",
    "pectoral_chest_involvement",
)

# Race/ethnicity is excluded by default. It can be loaded explicitly for
# fairness/subgroup analysis or a scientifically justified experiment.
SENSITIVE_CLINICAL_PREDICTOR_COLUMNS = ("race_ethnicity",)
_MISSING_CLINICAL_VALUES = {"", "NA", "N/A", "NC", "NP", "X", "NAN", "NONE"}


def base_path(target_path):
    """Resolve a Duke feature path while accepting existing absolute paths."""
    if os.path.isabs(target_path):
        return target_path
    return os.path.join(FEATURES_PATH, target_path)


def _clean_clinical_value(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip()
        if value.upper() in _MISSING_CLINICAL_VALUES:
            return pd.NA
    return value


def _categorize_oncotype_score(score, is_binary):
    """Map the recurrence score to the cutoffs used by the existing pipeline."""
    score = float(score)
    if is_binary:
        return 0 if score <= 18 else 1
    if score <= 18:
        return 0
    if score <= 31:
        return 1
    return 2


def read_useful_clinical_predictors(
    features_file=TARGETS_FILE_NAME,
    oncotype_only=False,
    include_oncotype_score=False,
    include_sensitive=False,
):
    """Load leakage-safe pretreatment clinical predictors from the Duke file.

    Treatment, pathologic response, recurrence, and follow-up variables are
    intentionally excluded because they occur after the pretreatment MRI and
    would leak outcome information. Except for age, the numeric clinical fields
    are category codes; fit their encoding/imputation only on the training fold.

    Args:
        features_file: Clinical workbook path. Relative paths are resolved under
            the configured Duke feature directory.
        oncotype_only: Keep only patients with a numeric Oncotype score.
        include_oncotype_score: Include the raw target score in the result.
        include_sensitive: Include race/ethnicity for an explicitly designed
            experiment or subgroup/fairness evaluation.

    Returns:
        A patient-level DataFrame with ``patientId`` and selected predictors.
    """
    data = pd.read_excel(
        base_path(features_file),
        sheet_name="Data",
        header=None,
        skiprows=3,
        usecols=_CLINICAL_EXCEL_COLUMNS,
        names=_CLINICAL_SOURCE_COLUMNS,
    )

    data = data.apply(lambda column: column.map(_clean_clinical_value))
    data = data.dropna(subset=["patientId"]).copy()
    data["patientId"] = data["patientId"].astype(str).str.strip()

    non_numeric_columns = {"patientId", "tumor_laterality"}
    for column in data.columns.difference(non_numeric_columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The source column is the number of days from the MRI date backwards to
    # birth, so it is negative for valid dates.
    data["age_at_diagnosis_years"] = -data["date_of_birth_days"] / 365.25

    if oncotype_only:
        data = data.loc[data["oncotype_score"].notna()].copy()

    selected_columns = ["patientId", *CLINICAL_PREDICTOR_COLUMNS]
    if include_sensitive:
        selected_columns.extend(SENSITIVE_CLINICAL_PREDICTOR_COLUMNS)
    if include_oncotype_score:
        selected_columns.append("oncotype_score")

    return data.loc[:, selected_columns].reset_index(drop=True)


def read_patient_id_for_oncotype_score_not_na(
    isBinary: bool = False,
    features_file=TARGETS_FILE_NAME,
):
    """Return patient IDs and binary or three-class Oncotype labels."""
    subset = read_useful_clinical_predictors(
        features_file=features_file,
        oncotype_only=True,
        include_oncotype_score=True,
    )[["patientId", "oncotype_score"]].copy()
    subset["oncotypeCategory"] = subset["oncotype_score"].apply(
        lambda score: _categorize_oncotype_score(score, isBinary)
    )
    return subset[["patientId", "oncotypeCategory"]]


def read_study_instance_for_patient_ids(patient_ids):
    images_metadata_file = base_path(IMAGES_METADATA)
    data = pd.read_excel(images_metadata_file, sheet_name="Metadata", header=0)

    data.rename(
        columns={
            "Patient ID": "patientId",
            "Study Instance UID": "studyId",
            "Series Instance UID": "seriesId",
        },
        inplace=True,
    )
    data["patientId"] = data["patientId"].astype(str)

    # Preserve all columns supplied by the patient-level table so this helper
    # can attach either labels alone or labels plus clinical predictors.
    return data.merge(patient_ids, on="patientId", how="inner")


def get_unique_studies():
    images_metadata_file = base_path(IMAGES_METADATA)
    data = pd.read_excel(images_metadata_file, sheet_name="Metadata", header=0)

    data.rename(
        columns={
            "Patient ID": "patientId",
            "Study Instance UID": "studyId",
            "Series Instance UID": "seriesId",
        },
        inplace=True,
    )

    return set(data.studyId)


def get_unique_study_instance_for_oncotype_score_as_not_na():
    patient_ids = read_patient_id_for_oncotype_score_not_na()
    return set(read_study_instance_for_patient_ids(patient_ids).studyId)


def get_oncotype_score_for_series(isBinary: bool):
    patient_ids = read_patient_id_for_oncotype_score_not_na(isBinary)
    return read_study_instance_for_patient_ids(patient_ids)


def get_oncotype_score_for_series_as_serie_and_label_df(
    num_of_samples=None,
    max_per_class=None,
    seed=None,
):
    data = get_oncotype_score_for_series(False)
    df = pd.DataFrame({"serie": data.seriesId, "label": data.oncotypeCategory})
    if num_of_samples is not None:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_per_class), random_state=seed)
        )

        if len(df) > num_of_samples:
            df = df.sample(n=num_of_samples, random_state=seed)

    return df


def get_oncotype_score_for_series_as_studyId_and_label_df(isBinary: bool):
    data = get_oncotype_score_for_series(isBinary)

    series_count = data.groupby("studyId")["seriesId"].nunique()
    # All configured workflows, including subtraction, share this study table.
    # Keep a common cohort that can form at least one subtraction. Studies
    # with more series remain valid because aggregation deterministically
    # selects the first MAX_SERIES_PER_STUDY entries below.
    valid_study_ids = series_count.loc[series_count >= 2].index
    filtered = data[data["studyId"].isin(valid_study_ids)]

    return (
        filtered.groupby("studyId")
        .agg(
            patientId=("patientId", "first"),
            series_ids=("seriesId", lambda x: list(x)[:MAX_SERIES_PER_STUDY]),
            label=("oncotypeCategory", "first"),
        )
        .reset_index()
    )


def get_oncotype_clinical_predictors_as_study_df(
    isBinary: bool = False,
    features_file=TARGETS_FILE_NAME,
    include_sensitive=False,
    include_oncotype_score=False,
):
    """Return one row per MRI study with series, label, and clinical features."""
    studies = get_oncotype_score_for_series_as_studyId_and_label_df(isBinary)
    clinical = read_useful_clinical_predictors(
        features_file=features_file,
        oncotype_only=True,
        include_oncotype_score=include_oncotype_score,
        include_sensitive=include_sensitive,
    )
    return studies.merge(clinical, on="patientId", how="inner", validate="many_to_one")

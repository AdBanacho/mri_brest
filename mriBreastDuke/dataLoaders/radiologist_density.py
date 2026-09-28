"""Optional BI-RADS density assessments for the tabular fusion branch."""

from collections import Counter

import pandas as pd


DENSITY_MODES = ("none", "majority", "radiologist_a", "radiologist_b", "radiologist_c")
_READERS = ("Radiologist A", "Radiologist B", "Radiologist C")
_CATEGORIES = {"a", "b", "c", "d"}
DENSITY_COLUMN = "radiologist_density"


def load_radiologist_density(path, mode):
    """Read the three ratings and select one category per exact patient ID.

    Majority requires at least two agreeing readers. Unrated or tied patients
    get an explicit `missing` category, never a guessed density or a mode
    imputation from other patients.
    """
    if mode not in DENSITY_MODES or mode == "none":
        raise ValueError(f"Select a density mode from {DENSITY_MODES[1:]}")
    table = pd.read_excel(path, sheet_name="Assessments", usecols="A:D", dtype=str)
    if list(table.columns) != ["Subject_ID", *_READERS]:
        raise ValueError(f"Unexpected density workbook columns: {list(table.columns)}")
    if table["Subject_ID"].isna().any():
        raise ValueError("Density workbook has a missing Subject_ID")
    table["patientId"] = table["Subject_ID"].str.strip()
    if table["patientId"].eq("").any() or table["patientId"].duplicated().any():
        raise ValueError("Density workbook contains blank or duplicate Subject_ID values")
    for reader in _READERS:
        table[reader] = table[reader].str.strip().str.lower()
        invalid = table[reader].notna() & ~table[reader].isin(_CATEGORIES)
        if invalid.any():
            raise ValueError(f"Invalid {reader} density category for {table.loc[invalid, 'patientId'].tolist()[:5]}")

    if mode == "majority":
        def vote(row):
            counts = Counter(value for value in row if pd.notna(value))
            return next((category for category, count in counts.items() if count >= 2), "missing")

        selected = table[list(_READERS)].apply(vote, axis=1)
    else:
        selected = table["Radiologist " + mode[-1].upper()]
    return pd.DataFrame({"patientId": table["patientId"], DENSITY_COLUMN: selected.fillna("missing")})


def merge_radiologist_density(studies, path, mode):
    """Left join by patient; preserve cohort and mark unmatched rows missing."""
    if "patientId" not in studies.columns:
        raise ValueError("Studies table must contain patientId for density matching")
    density = load_radiologist_density(path, mode)
    prepared = studies.copy()
    prepared["patientId"] = prepared["patientId"].astype(str).str.strip()
    merged = prepared.merge(density, on="patientId", how="left", validate="many_to_one")
    matched = merged[DENSITY_COLUMN].notna().sum()
    if not matched:
        raise ValueError("Density Subject_ID did not match any modeled patientId")
    merged[DENSITY_COLUMN] = merged[DENSITY_COLUMN].fillna("missing")
    print(
        f"[DENSITY] mode={mode}; rated={int((merged[DENSITY_COLUMN] != 'missing').sum())}"
        f"/{len(merged)} studies; unmatched_or_unrated={int((merged[DENSITY_COLUMN] == 'missing').sum())}",
        flush=True,
    )
    return merged, DENSITY_COLUMN

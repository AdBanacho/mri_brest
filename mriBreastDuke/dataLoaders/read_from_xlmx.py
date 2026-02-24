import os
import pandas as pd

from mriBreastDuke.constants import DUKE_PATH, FEATURES_PATH, TARGETS_FILE_NAME, IMAGES_METADATA


def base_path(target_path):
    return os.path.join(DUKE_PATH, FEATURES_PATH, target_path)


def read_patient_id_for_oncotype_score_not_na():
    features_file = base_path(TARGETS_FILE_NAME)
    data = pd.read_excel(features_file, sheet_name="Data", header=[0, 1])

    data.columns = [
        ' '.join([str(x) for x in col if str(x) != 'nan']).strip()
        for col in data.columns.values
    ]

    data = data.drop(index=[0, 1]).reset_index(drop=True)

    subset = data.loc[
        data['Tumor Characteristics Oncotype score'].notna(),
        ['Patient Information Patient ID', 'Tumor Characteristics Oncotype score']
    ].copy()

    subset.rename(columns={"Patient Information Patient ID": "patientId"}, inplace=True)
    subset['patientId'] = subset['patientId'].astype(str)

    def categorize(score): # 0-18, 19-31, <31
        score = float(score)
        if score <= 18:
            return 0
        elif score <= 31:
            return 1
        else:
            return 2

    subset['oncotypeCategory'] = subset['Tumor Characteristics Oncotype score'].apply(categorize)

    return subset[['patientId', 'oncotypeCategory']]


def read_study_instance_for_patient_ids(patient_ids):
    images_metadata_file = base_path(IMAGES_METADATA)
    data = pd.read_excel(images_metadata_file, sheet_name="Metadata", header=0)

    data.rename(columns={"Patient ID": "patientId",
                         "Study Instance UID": "studyId",
                         "Series Instance UID": "seriesId"}, inplace=True)

    return data.merge(patient_ids, on="patientId", how="inner")[['patientId', 'studyId', 'seriesId', 'oncotypeCategory']]


def get_unique_studies():
    images_metadata_file = base_path(IMAGES_METADATA)
    data = pd.read_excel(images_metadata_file, sheet_name="Metadata", header=0)

    data.rename(columns={"Patient ID": "patientId",
                         "Study Instance UID": "studyId",
                         "Series Instance UID": "seriesId"}, inplace=True)

    return set(data.studyId)


def get_unique_study_instance_for_oncotype_score_as_not_na():
    patient_ids = read_patient_id_for_oncotype_score_not_na()
    return set(read_study_instance_for_patient_ids(patient_ids).studyId)


def get_oncotype_score_for_series():
    patient_ids = read_patient_id_for_oncotype_score_not_na()
    return read_study_instance_for_patient_ids(patient_ids)


def get_oncotype_score_for_series_as_serie_and_label_df(num_of_samples=None, max_per_class=None, seed=None):
    data = get_oncotype_score_for_series()
    df = pd.DataFrame({
        "serie": data.seriesId,
        "label": data.oncotypeCategory
    })
    if num_of_samples is not None:
        df = df.groupby("label", group_keys=False)\
            .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=seed))

        if len(df) > num_of_samples:
            df = df.sample(n=num_of_samples, random_state=seed)

    return df

def get_oncotype_score_for_series_as_studyId_and_label_df():
    data = get_oncotype_score_for_series()

    # Find studyIds that do NOT have exactly 4 unique series
    valid_study_ids = (
        data.groupby("studyId")["seriesId"]
        .nunique()
        .loc[lambda x: x != 4]
        .index
    )

    # Reduce df to match those studyIds
    df = pd.DataFrame({
        "serie": data.studyId,
        "label": data.oncotypeCategory
    }).loc[lambda x: x["serie"].isin(valid_study_ids)]

    # Grouped data, consistent with df
    grouped_by_study = data[data["studyId"].isin(valid_study_ids)].groupby("studyId")

    return df, grouped_by_study

import os
import pandas as pd
from pathlib import Path

THIS_FILE = Path(__file__).resolve()

# project root = parent folder of mri_breast_duke and mama_mia
PROJECT_ROOT = THIS_FILE.parents[1]

FOLDER_PATH = "mri_breast_duke"
FEATURES_PATH = "features"
IMAGES_METADATA = "Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx"
TARGETS_FILE_NAME = "Clinical_and_Other_Features.xlsx"


def base_path(target_path):
    return os.path.join(PROJECT_ROOT,  FOLDER_PATH, FEATURES_PATH, target_path)


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

    def categorize(score):
        score = float(score)
        if score < 5:
            return 0
        elif score < 20:
            return 1
        elif score < 50:
            return 2
        else:
            return 3

    subset['oncotypeCategory'] = subset['Tumor Characteristics Oncotype score'].apply(categorize)

    return subset[['patientId', 'oncotypeCategory']]


def read_study_instance_for_patient_ids(patient_ids):
    images_metadata_file = base_path(IMAGES_METADATA)
    data = pd.read_excel(images_metadata_file, sheet_name="Metadata", header=0)

    data.rename(columns={"Patient ID": "patientId",
                         "Study Instance UID": "studyId",
                         "Series Instance UID": "seriesId"}, inplace=True)

    return data.merge(patient_ids, on="patientId", how="inner")[['patientId', 'studyId', 'seriesId', 'oncotypeCategory']]


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

import pandas as pd
from mri_breast_duke.readFromXlmx import get_oncotype_score_for_series

FEATURES_PATH = "features/"
TARGETS_FILE_NAME = "41597_2025_4707_MOESM2_ESM.xlsx"


def read_series_uids():
    images_metadata_file = FEATURES_PATH + TARGETS_FILE_NAME
    data = pd.read_excel(images_metadata_file, sheet_name="dataset_info", header=0)

    return data.tcia_series_uid


def get_mamamia_images_for_oncotype():
    duke_oncotype_series = get_oncotype_score_for_series().seriesId
    mama_mia_series = read_series_uids()

    return list(set(duke_oncotype_series) & set(mama_mia_series))


print(len(get_mamamia_images_for_oncotype()))
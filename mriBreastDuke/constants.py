import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = "mriBreastDuke"
DUKE_PATH = os.path.join(PROJECT_ROOT, PROJECT_PATH)

SEED = 42

IMAGES_PATH = os.path.join(DUKE_PATH, "images")
DCM_PATH = os.path.join(IMAGES_PATH, "tciaDownload")
NIFTI_PATH = os.path.join(IMAGES_PATH, "tciaNifti")
PREPARED_TO_TRAIN_PATH = os.path.join(IMAGES_PATH, "preparedToTrain")

DUKE_CANCER_MRI_COLLECTION = "Duke-Breast-Cancer-MRI"

FEATURES_PATH = os.path.join(DUKE_PATH, "features")
IMAGES_METADATA = os.path.join(FEATURES_PATH, "Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx")
TARGETS_FILE_NAME = os.path.join(FEATURES_PATH, "Clinical_and_Other_Features.xlsx")
SIZE_CACHE_PATH = lambda x: os.path.join(FEATURES_PATH, x + "_volume_sizes.csv")

LIGHTING_LOGS = "lightning_logs"

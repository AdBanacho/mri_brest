import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = "mriBreastDuke"
DUKE_PATH = os.path.join(PROJECT_ROOT, PROJECT_PATH)
HOME_NET = "/net"
STORAGE = "storage"
PR3 = "pr3"
PLGRID = "plgrid"
PLGGSIWOMIPIAN = "plggsiwomipipan"
ABANACHO = "aBanacho_duke_oncotype"
STORAGE_PATH = os.path.join(HOME_NET, STORAGE, PR3, PLGRID, PLGGSIWOMIPIAN, ABANACHO)

SCRATCH = "scratch"
HSCRA = "hscra"
PLGABANACHO = "plgabanacho"
IMAGES_HELIOS_PATH = os.path.join(HOME_NET, SCRATCH, HSCRA, PLGRID, PLGABANACHO)

SEED = 42
MAX_SERIES_PER_STUDY = 5

IMAGES_PATH = os.path.join(STORAGE_PATH, "images")
DCM_PATH = os.path.join(STORAGE_PATH, "tciaDownload")
NIFTI_PATH = os.path.join(STORAGE_PATH, "tciaNifti")
PREPARED_TO_TRAIN_PATH = os.path.join(IMAGES_HELIOS_PATH, "preparedToTrain")
SUBTRACTION_PATH = os.path.join(IMAGES_HELIOS_PATH, "preparedToTrainSubtraction")

DUKE_CANCER_MRI_COLLECTION = "Duke-Breast-Cancer-MRI"

FEATURES_PATH = os.path.join(DUKE_PATH, "features")
IMAGES_METADATA = os.path.join(FEATURES_PATH, "Duke-Breast-Cancer-MRI_v2_20220609-nbia-digest.xlsx")
TARGETS_FILE_NAME = os.path.join(FEATURES_PATH, "Clinical_and_Other_Features.xlsx")
SIZE_CACHE_PATH = lambda x: os.path.join(FEATURES_PATH, x + "_volume_sizes.csv")

LIGHTING_LOGS = "lightning_logs"
CHECKPOINTS_PATH = os.path.join(IMAGES_HELIOS_PATH, "check_points")

VALIDATION_CHART_PATH = "validation_charts"

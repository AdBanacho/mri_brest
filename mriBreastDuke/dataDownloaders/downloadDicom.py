from tcia_utils import nbia
from tqdm import tqdm

from mriBreastDuke.constants import DUKE_CANCER_MRI_COLLECTION
from mriBreastDuke.dataLoaders import get_unique_study_instance_for_oncotype_score_as_not_na

uuids = get_unique_study_instance_for_oncotype_score_as_not_na()
for id in tqdm(uuids):
    data = nbia.getSeries(collection=DUKE_CANCER_MRI_COLLECTION, studyUid=id)
    nbia.downloadSeries(data)

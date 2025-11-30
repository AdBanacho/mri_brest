from readFromXlmx import get_unique_study_instance_for_oncotype_score_as_not_na
from tcia_utils import nbia
from tqdm import tqdm

uuids = get_unique_study_instance_for_oncotype_score_as_not_na()
for id in tqdm(uuids):
    data = nbia.getSeries(collection="Duke-Breast-Cancer-MRI", studyUid=id)
    nbia.downloadSeries(data)

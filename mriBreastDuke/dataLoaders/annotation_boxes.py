"""Patient-level Duke boxes for training supervision and retrospective audits.

Workbook indices are interpreted as zero-based, inclusive DICOM row/column/slice
indices. SimpleITK NIfTI voxel axes are column, row, slice; no reorientation is
performed by the current loader. The workbook has no series UID or affine, so
only studies with one patient study and identical series grids are eligible.
"""

from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from mriBreastDuke.constants import ANNOTATION_BOXES_FILE_NAME, NIFTI_PATH


COLUMNS = (
    "Patient ID", "Start Row", "End Row", "Start Column", "End Column",
    "Start Slice", "End Slice",
)


def load_annotation_boxes(path=ANNOTATION_BOXES_FILE_NAME):
    """Return patient -> half-open (column, row, slice) box; reject bad rows."""
    data = pd.read_excel(path, sheet_name="Sheet1")
    if tuple(data.columns) != COLUMNS:
        raise ValueError(f"Unexpected annotation columns: {list(data.columns)}")
    boxes = {}
    for row_number, row in enumerate(data.itertuples(index=False, name=None), start=2):
        patient = row[0]
        if not isinstance(patient, str) or not patient.strip():
            raise ValueError(f"Annotation row {row_number}: missing Patient ID")
        patient = patient.strip()
        if patient in boxes:
            raise ValueError(f"Duplicate annotation for {patient} (row {row_number})")
        values = row[1:]
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
            or not np.isfinite(v) or v < 0 or int(v) != v
            for v in values
        ):
            raise ValueError(f"Annotation row {row_number}: nonnegative integer coordinates required")
        r0, r1, c0, c1, s0, s1 = map(int, values)
        if r0 > r1 or c0 > c1 or s0 > s1:
            raise ValueError(f"Annotation row {row_number}: reversed box bounds")
        boxes[patient] = ((c0, r0, s0), (c1 + 1, r1 + 1, s1 + 1))
    return boxes


def annotation_coverage(studies, boxes, image_root=NIFTI_PATH):
    """Audit all studies and return only boxes safe to apply to every series.

    Missing annotations, ambiguous patient studies and incompatible series are
    reported and left as full images. Out-of-volume coordinates are an error.
    """
    if not {"patientId", "studyId", "series_ids"}.issubset(studies.columns):
        raise ValueError("Annotation matching requires patientId, studyId and series_ids")
    counts = Counter(studies["patientId"].astype(str).str.strip())
    if studies["studyId"].duplicated().any():
        raise ValueError("Duplicate studyId rows make annotation matching ambiguous")
    matched = {}
    report = []
    for row in studies.itertuples(index=False):
        patient = str(row.patientId).strip()
        status = "eligible"
        box = boxes.get(patient)
        if box is None:
            status = "missing_box"
        elif counts[patient] != 1:
            status = "ambiguous_patient_studies"
        elif not isinstance(row.series_ids, (list, tuple)) or not row.series_ids:
            status = "missing_series"
        else:
            reference = None
            for series_id in row.series_ids:
                image = nib.load(str(Path(image_root) / f"{series_id}.nii.gz"))
                shape = tuple(image.shape)
                if len(shape) != 3:
                    status = "incompatible_series_geometry"
                    break
                if any(end > size for end, size in zip(box[1], shape)):
                    raise ValueError(
                        f"Annotation for {patient} exceeds {series_id} image shape {shape}: {box}"
                    )
                geometry = (shape, np.asarray(image.affine))
                if reference is not None and (
                    shape != reference[0]
                    or not np.allclose(geometry[1], reference[1], rtol=1e-4, atol=1e-3)
                ):
                    status = "incompatible_series_geometry"
                    break
                reference = geometry
            if status == "eligible":
                matched[row.studyId] = (box, reference[0])
        report.append({"patientId": patient, "studyId": row.studyId, "status": status})
    unused = sorted(set(boxes).difference(studies["patientId"].astype(str).str.strip()))
    report.extend({"patientId": patient, "studyId": "", "status": "unmatched_annotation"}
                  for patient in unused)
    return matched, pd.DataFrame(report, columns=["patientId", "studyId", "status"])


def resize_box(box, source_shape, target_shape):
    """Map half-open voxel edges through the load-time axis-aligned resize."""
    starts, ends = box
    starts = tuple(int(np.floor(a * dst / src)) for a, src, dst in zip(starts, source_shape, target_shape))
    ends = tuple(int(np.ceil(b * dst / src)) for b, src, dst in zip(ends, source_shape, target_shape))
    return starts, ends

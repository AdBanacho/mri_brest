# Conservative clinical features for MRI and Oncotype score modeling

The default `clinical` branch now uses only the following columns from
`Clinical_and_Other_Features.xlsx` (`Data` sheet):

| Predictor | Workbook source | Availability assumption |
| --- | --- | --- |
| `age_at_diagnosis_years` | T, date of birth relative to diagnosis | Known by the MRI visit |
| `menopause` | U, menopause at diagnosis | Known by the MRI visit |
| `tumor_laterality` | AK, side of cancer | Known from the initial examination/imaging; new predictor |
| `multicentric_multifocal` | AW, MRI finding | Initial MRI report available |
| `contralateral_breast_involvement` | AX, MRI finding | Initial MRI report available |
| `suspicious_lymph_nodes` | AY, MRI finding | Initial MRI report available |
| `skin_nipple_involvement` | AZ, MRI finding | Initial MRI report available |
| `pectoral_chest_involvement` | BA, MRI finding | Initial MRI report available |

The `oncotype_score` column AB is read **solely as a target**, for restricting
the retrospectively labeled cohort and computing the existing class labels.
It is never included in the predictor set. `race_ethnicity` (V) remains
excluded by default and can be enabled explicitly with `--include_sensitive`.

Removed from the default predictors: ER, PR, HER2, molecular subtype (X–AA),
metastatic status (W), staging (AC–AE), tubule/nuclear/mitotic/Nottingham
grades (AF–AI), and histologic type (AJ). Grade and PR in particular can
predict the Recurrence Score from conventional pathology; the receptor fields
also determine assay eligibility. Their correlation with a genomic score is
**not itself label leakage** when measured before testing. Excluding them
here defines a conservative MRI-focused comparison and avoids the uncertain
recording dates in this retrospective workbook. ER/HER2 and stage should be
evaluated separately as *eligibility criteria* for a clinically defined
cohort, using information verified to precede the assay. The current cohort
selection is unchanged by this feature update.

Surgery, systemic therapy, response, recurrence and follow-up columns (BB
onward, except BIRADS) remain excluded: some are determined after the assay
or can reflect decisions made using its result. BIRADS mammography and US
fields were considered, but in the 261 patients with numeric scores, their
individual coverage is generally very low (roughly 1–13% for sampled fields),
and this file has no examination dates to prove pretest availability. They
are not added to the default model. Scanner settings and diagnosis-to-MRI
interval are also omitted to avoid care-pathway and acquisition proxies.

The workbook does not record Oncotype order/result dates or feature
measurement dates. Before a prospective claim, verify for each patient that
the MRI and every included finding were available **before** the assay result.
This code change alone cannot establish temporal validity. If the intended
decision point is before the MRI report, remove the five reported MRI
findings too and use only variables known at that earlier point.

Clinical experiments now include `_clinical-safe-v2` in the shared training
and validation checkpoint name; this prevents an older tabular model fitted
on the previous feature list from being validated as though it used this one.
Training and validation commands retain their existing options, for example:

```bash
python -m mriBreastDuke.configurable_imaging_features_fusion_workflow --feature_groups clinical
python -m mriBreastDuke.validate_configurable_imaging_features_fusion --feature_groups clinical
```

Sources: [ASCO biomarker guideline](https://ascopubs.org/doi/full/10.1200/JCO.22.00069),
[Oncotype DX test eligibility and tissue source](https://www.exactsciences.com/cancer-testing/oncotype-dx-breast-recurrence-score-providers),
and [published grade/PR association with the score](https://pubmed.ncbi.nlm.nih.gov/21369717/).

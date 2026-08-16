"""Cross-validated MRI + clinical/radiomics late-fusion training workflow."""

import argparse
import math
from numbers import Number

from monai.networks.nets import DenseNet121, resnet18
import pandas as pd
import pytorch_lightning as pl

from mriBreastDuke.classificators import (
    FusionClassifier,
    NiftiClassifier,
    Simple3DFCN,
)
from mriBreastDuke.constants import SEED
from mriBreastDuke.dataLoaders import (
    CLINICAL_PREDICTOR_COLUMNS,
    SENSITIVE_CLINICAL_PREDICTOR_COLUMNS,
    get_oncotype_clinical_predictors_as_study_df,
    SUBTRACTION_MODES,
    SUBTRACTION_NONE,
    get_input_channels,
)
from mriBreastDuke.n_fold_cv_run import run_5fold_cv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a late-fusion DCE-MRI and tabular Oncotype classifier."
    )
    parser.add_argument(
        "--backbone",
        choices=("fcn", "densenet121", "resnet18"),
        default="resnet18",
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--is_binary_classification", action="store_true")
    parser.add_argument("--include_sensitive", action="store_true")
    parser.add_argument("--radiomics_csv", default=None)
    parser.add_argument("--radiomics_key", default="studyId")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sensitivity_lambda", type=float, default=0.3)
    parser.add_argument("--positive_boost", type=float, default=1.0)
    parser.add_argument("--image_embedding_dim", type=int, default=128)
    parser.add_argument("--clinical_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument(
        "--subtraction_mode",
        choices=SUBTRACTION_MODES,
        default=SUBTRACTION_NONE,
    )
    return parser.parse_args()


def _merge_radiomics(studies, csv_path, merge_key):
    radiomics = pd.read_csv(csv_path)
    if merge_key not in studies.columns or merge_key not in radiomics.columns:
        raise ValueError(
            f"Radiomics merge key '{merge_key}' must be present in both tables."
        )
    if radiomics[merge_key].duplicated().any():
        raise ValueError(f"Radiomics table contains duplicate '{merge_key}' values.")

    reserved = {
        merge_key,
        "patientId",
        "label",
        "oncotype_score",
        "series_ids",
        "lesion_mask_path",
    }
    categorical_features = [
        column
        for column in radiomics.columns
        if column.endswith("kinetic_curve_type")
        and column not in reserved
        and column not in studies.columns
    ]
    numeric_features = [
        column
        for column in radiomics.select_dtypes(include="number").columns
        if column not in reserved
        and column not in studies.columns
        and not column.endswith("kinetic_curve_type_code")
    ]
    feature_columns = [*numeric_features, *categorical_features]
    if not feature_columns:
        raise ValueError("No usable radiomics columns were found in radiomics_csv.")

    merged = studies.merge(
        radiomics[[merge_key, *feature_columns]],
        on=merge_key,
        how="left",
        validate="one_to_one",
    )
    matched = merged[feature_columns].notna().any(axis=1).sum()
    if matched == 0:
        raise ValueError("The radiomics table did not match any MRI studies.")
    print(
        f"[FUSION] Added {len(feature_columns)} radiomics features; "
        f"matched {matched}/{len(merged)} studies.",
        flush=True,
    )
    return merged, numeric_features, categorical_features


def _make_image_encoder(backbone, in_channels, embedding_dim):
    if backbone == "fcn":
        return Simple3DFCN(
            num_classes=embedding_dim,
            in_channels=in_channels,
        )
    if backbone == "densenet121":
        return DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=embedding_dim,
        )
    return resnet18(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=embedding_dim,
    )


def main():
    args = parse_args()
    studies = get_oncotype_clinical_predictors_as_study_df(
        isBinary=args.is_binary_classification,
        include_sensitive=args.include_sensitive,
        include_oncotype_score=False,
    )
    num_classes = int(studies["label"].nunique())
    in_channels = get_input_channels(args.subtraction_mode)

    continuous_columns = ["age_at_diagnosis_years"]
    categorical_columns = [
        column
        for column in CLINICAL_PREDICTOR_COLUMNS
        if column != "age_at_diagnosis_years"
    ]
    if args.include_sensitive:
        categorical_columns.extend(SENSITIVE_CLINICAL_PREDICTOR_COLUMNS)

    radiomics_suffix = ""
    if args.radiomics_csv:
        studies, radiomics_columns, radiomics_categorical_columns = _merge_radiomics(
            studies,
            args.radiomics_csv,
            args.radiomics_key,
        )
        continuous_columns.extend(radiomics_columns)
        categorical_columns.extend(radiomics_categorical_columns)
        radiomics_suffix = (
            f"_rad{len(radiomics_columns) + len(radiomics_categorical_columns)}"
        )

    model_name = (
        f"Fusion_{args.backbone}"
        f"_lr_{args.lr:.0e}"
        f"_img{args.image_embedding_dim}"
        f"_clin{args.clinical_hidden_dim}"
        f"_fuse{args.fusion_hidden_dim}"
        f"_drop{args.dropout:g}"
        f"_{args.subtraction_mode}"
        f"{radiomics_suffix}"
    )

    def make_model(class_weights, clinical_input_dim):
        image_encoder = _make_image_encoder(
            args.backbone,
            in_channels,
            args.image_embedding_dim,
        )
        fusion_model = FusionClassifier(
            image_encoder=image_encoder,
            image_feature_dim=args.image_embedding_dim,
            clinical_input_dim=clinical_input_dim,
            num_classes=num_classes,
            clinical_hidden_dim=args.clinical_hidden_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
        )
        return NiftiClassifier(
            fusion_model,
            num_classes=num_classes,
            lr=args.lr,
            class_weights=class_weights,
            sensitivity_lambda=args.sensitivity_lambda,
        )

    metrics_per_fold = run_5fold_cv(
        df=studies,
        model_name=model_name,
        make_model=make_model,
        epoch=args.epoch,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_boost=args.positive_boost,
        subtraction_mode=args.subtraction_mode,
        clinical_continuous_columns=continuous_columns,
        clinical_categorical_columns=categorical_columns,
        group_column="patientId",
    )

    print("\n========== Fusion CV Summary ==========")
    keys = sorted({key for metrics in metrics_per_fold for key in metrics})
    for key in keys:
        values = [metrics[key] for metrics in metrics_per_fold if key in metrics]
        numeric_values = [
            value
            for value in values
            if isinstance(value, Number)
            and not (isinstance(value, float) and math.isnan(value))
        ]
        if not numeric_values:
            continue
        mean = sum(numeric_values) / len(numeric_values)
        std = (
            sum((value - mean) ** 2 for value in numeric_values)
            / len(numeric_values)
        ) ** 0.5
        print(f"{key}: mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

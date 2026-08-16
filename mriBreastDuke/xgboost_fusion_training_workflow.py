"""MRI + XGBoost decision fusion; no clinical multilayer perceptron is used."""

import argparse
import math
from numbers import Number

from monai.networks.nets import DenseNet121, resnet18
import pandas as pd
import pytorch_lightning as pl

from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
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
        description="Patient-grouped MRI + XGBoost probability fusion."
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
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="MRI probability weight; XGBoost weight is 1 - fusion_alpha.",
    )
    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=3)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_min_child_weight", type=float, default=2.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_n_jobs", type=int, default=8)
    parser.add_argument(
        "--subtraction_mode",
        choices=SUBTRACTION_MODES,
        default=SUBTRACTION_NONE,
    )
    return parser.parse_args()


def merge_radiomics(studies, csv_path, merge_key):
    """Attach numeric radiomics and categorical kinetic curve type by study."""
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
    matched = int(merged[feature_columns].notna().any(axis=1).sum())
    if matched == 0:
        raise ValueError("The radiomics table did not match any MRI studies.")
    print(
        f"[XGBOOST FUSION] Added {len(feature_columns)} radiomics features; "
        f"matched {matched}/{len(merged)} studies.",
        flush=True,
    )
    return merged, numeric_features, categorical_features


def make_image_model(backbone, in_channels, num_classes):
    """Build an image-only classifier that produces class logits."""
    if backbone == "fcn":
        return Simple3DFCN(num_classes=num_classes, in_channels=in_channels)
    if backbone == "densenet121":
        return DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
        )
    return resnet18(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=num_classes,
    )


def make_xgboost_model(args, num_classes):
    """Construct the official ``xgboost.XGBClassifier`` tabular branch."""
    parameters = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "min_child_weight": args.xgb_min_child_weight,
        "reg_lambda": args.xgb_reg_lambda,
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": args.xgb_n_jobs,
        "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
        "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
    }
    if num_classes > 2:
        parameters["num_class"] = num_classes
    return XGBClassifier(**parameters)


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
        studies, numeric_radiomics, categorical_radiomics = merge_radiomics(
            studies,
            args.radiomics_csv,
            args.radiomics_key,
        )
        continuous_columns.extend(numeric_radiomics)
        categorical_columns.extend(categorical_radiomics)
        radiomics_suffix = f"_rad{len(numeric_radiomics) + len(categorical_radiomics)}"

    model_name = (
        f"XGBoostDecisionFusion_{args.backbone}"
        f"_lr_{args.lr:.0e}"
        f"_alpha{args.fusion_alpha:g}"
        f"_trees{args.xgb_n_estimators}"
        f"_{args.subtraction_mode}"
        f"{radiomics_suffix}"
    )

    def image_model_factory(class_weights):
        return NiftiClassifier(
            make_image_model(args.backbone, in_channels, num_classes),
            num_classes=num_classes,
            lr=args.lr,
            class_weights=class_weights,
            sensitivity_lambda=args.sensitivity_lambda,
        )

    def xgboost_model_factory():
        return make_xgboost_model(args, num_classes)

    metrics_per_fold = run_5fold_cv(
        df=studies,
        model_name=model_name,
        make_model=image_model_factory,
        epoch=args.epoch,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_boost=args.positive_boost,
        subtraction_mode=args.subtraction_mode,
        clinical_continuous_columns=continuous_columns,
        clinical_categorical_columns=categorical_columns,
        group_column="patientId",
        tabular_model_factory=xgboost_model_factory,
        fusion_alpha=args.fusion_alpha,
    )

    print("\n========== MRI + XGBoost CV Summary ==========")
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

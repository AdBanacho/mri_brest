"""Standalone configurable MRI fusion workflow with cached feature extraction."""

import argparse
import math
from numbers import Number

from monai.networks.nets import DenseNet121, resnet18
import pandas as pd
import pytorch_lightning as pl
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.constants import NIFTI_PATH, SEED
from mriBreastDuke.dataLoaders import (
    CLINICAL_PREDICTOR_COLUMNS,
    SENSITIVE_CLINICAL_PREDICTOR_COLUMNS,
    get_oncotype_clinical_predictors_as_study_df,
    SUBTRACTION_MODES,
    SUBTRACTION_NONE,
    get_input_channels,
    ensure_radiomics_cache
)
from mriBreastDuke.n_fold_cv_run import run_5fold_cv


FEATURE_GROUPS = ("clinical", "kinetic", "morphology", "heterogeneity")
TABULAR_MODELS = ("xgboost", "mlp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure MRI and clinical/radiomics probability fusion."
    )
    parser.add_argument(
        "--mri_model",
        choices=("fcn", "densenet121", "resnet18"),
        default="densenet121",
    )
    parser.add_argument(
        "--subtraction_mode",
        choices=SUBTRACTION_MODES,
        default=SUBTRACTION_NONE,
    )
    parser.add_argument(
        "--feature_groups",
        nargs="+",
        choices=FEATURE_GROUPS,
        default=["clinical"],
    )
    parser.add_argument(
        "--feature_model",
        choices=TABULAR_MODELS,
        default="xgboost",
    )
    parser.add_argument(
        "--radiomics_csv",
        default="duke_mri_features.csv",
        help="Feature-cache path. It is created automatically when missing.",
    )
    parser.add_argument("--radiomics_key", default="studyId")
    parser.add_argument("--image_root", default=NIFTI_PATH)
    parser.add_argument("--lesion_masks_csv", default=None)
    parser.add_argument("--lesion_mask_root", default=None)
    parser.add_argument("--lesion_mask_column", default="lesion_mask_path")
    parser.add_argument("--lesion_mask_suffix", default=".nii.gz")
    parser.add_argument("--registered_mask_root", default=None)
    parser.add_argument("--mask_transform_column", default=None)
    parser.add_argument("--radiomics_lock_timeout", type=float, default=21600)
    parser.add_argument("--include_sensitive", action="store_true")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--positive_boost", type=float, default=1.0)
    parser.add_argument("--sensitivity_lambda", type=float, default=0.3)
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="MRI probability weight; tabular model weight is 1 - alpha.",
    )

    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=3)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_n_jobs", type=int, default=8)

    parser.add_argument("--mlp_hidden_layers", default="128,64")
    parser.add_argument("--mlp_alpha", type=float, default=1e-4)
    parser.add_argument("--mlp_learning_rate", type=float, default=1e-3)
    parser.add_argument("--mlp_max_iter", type=int, default=500)
    return parser.parse_args()


def _selected_radiomics_columns(radiomics, selected_groups, reserved):
    selected_columns = []
    categorical_columns = []
    for group in selected_groups:
        prefix = f"{group}_"
        group_columns = [
            column
            for column in radiomics.columns
            if column.startswith(prefix)
            and column not in reserved
            and column != "heterogeneity_source"
        ]
        for column in group_columns:
            if column.endswith("kinetic_curve_type"):
                categorical_columns.append(column)
            elif column.endswith("kinetic_curve_type_code"):
                continue
            elif pd.api.types.is_numeric_dtype(radiomics[column]):
                selected_columns.append(column)
    return selected_columns, categorical_columns


def merge_selected_radiomics(studies, csv_path, merge_key, selected_groups):
    """Merge only the requested kinetic, morphology, or heterogeneity groups."""
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
        "registered_lesion_mask_path",
    }
    numeric_columns, categorical_columns = _selected_radiomics_columns(
        radiomics,
        selected_groups,
        reserved,
    )
    feature_columns = [*numeric_columns, *categorical_columns]
    if not feature_columns:
        raise ValueError(
            "None of the requested radiomics feature groups were found in the CSV: "
            f"{sorted(selected_groups)}"
        )

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
        f"[FEATURES] groups={sorted(selected_groups)} "
        f"columns={len(feature_columns)} matched={matched}/{len(merged)}",
        flush=True,
    )
    return merged, numeric_columns, categorical_columns


def make_mri_model(name, in_channels, num_classes):
    if name == "fcn":
        return Simple3DFCN(num_classes=num_classes, in_channels=in_channels)
    if name == "densenet121":
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


def make_tabular_model(args, num_classes):
    if args.feature_model == "xgboost":
        parameters = {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": args.xgb_n_jobs,
            "objective": (
                "binary:logistic" if num_classes == 2 else "multi:softprob"
            ),
            "eval_metric": "logloss" if num_classes == 2 else "mlogloss",
        }
        if num_classes > 2:
            parameters["num_class"] = num_classes
        return XGBClassifier(**parameters)

    hidden_layers = tuple(
        int(width.strip())
        for width in args.mlp_hidden_layers.split(",")
        if width.strip()
    )
    if not hidden_layers:
        raise ValueError("mlp_hidden_layers must contain at least one positive width.")
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=args.mlp_alpha,
        learning_rate_init=args.mlp_learning_rate,
        max_iter=args.mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEED,
    )


def main():
    args = parse_args()
    selected_groups = tuple(dict.fromkeys(args.feature_groups))
    radiomics_groups = tuple(
        group for group in selected_groups if group != "clinical"
    )
    studies = get_oncotype_clinical_predictors_as_study_df(
        isBinary=True,
        include_sensitive=args.include_sensitive,
        include_oncotype_score=False,
    )
    num_classes = int(studies["label"].nunique())
    in_channels = get_input_channels(args.subtraction_mode)

    continuous_columns = []
    categorical_columns = []
    if "clinical" in selected_groups:
        continuous_columns.append("age_at_diagnosis_years")
        categorical_columns.extend(
            column
            for column in CLINICAL_PREDICTOR_COLUMNS
            if column != "age_at_diagnosis_years"
        )
        if args.include_sensitive:
            categorical_columns.extend(SENSITIVE_CLINICAL_PREDICTOR_COLUMNS)

    if radiomics_groups:
        radiomics_csv = ensure_radiomics_cache(
            studies,
            cache_path=args.radiomics_csv,
            image_root=args.image_root,
            merge_key=args.radiomics_key,
            mask_path_column=args.lesion_mask_column,
            lesion_masks_csv=args.lesion_masks_csv,
            lesion_mask_root=args.lesion_mask_root,
            lesion_mask_suffix=args.lesion_mask_suffix,
            registered_mask_root=args.registered_mask_root,
            mask_transform_column=args.mask_transform_column,
            lock_timeout=args.radiomics_lock_timeout,
        )
        studies, numeric_radiomics, categorical_radiomics = merge_selected_radiomics(
            studies,
            radiomics_csv,
            args.radiomics_key,
            radiomics_groups,
        )
        continuous_columns.extend(numeric_radiomics)
        categorical_columns.extend(categorical_radiomics)

    feature_tag = "-".join(selected_groups)
    model_name = (
        f"ConfigFusion_{args.mri_model}_{args.subtraction_mode}"
        f"_{feature_tag}_{args.feature_model}"
        f"_lr{args.lr:.0e}_sens{args.sensitivity_lambda:g}"
        f"_boost{args.positive_boost:g}_bs{args.batch_size}"
    )

    def image_model_factory(class_weights):
        image_model = make_mri_model(args.mri_model, in_channels, num_classes)
        return NiftiClassifier(
            image_model,
            num_classes=num_classes,
            lr=args.lr,
            class_weights=class_weights,
            sensitivity_lambda=args.sensitivity_lambda,
        )

    def tabular_model_factory():
        return make_tabular_model(args, num_classes)

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
        tabular_model_factory=tabular_model_factory,
        tabular_model_name=args.feature_model,
        fusion_alpha=args.fusion_alpha,
    )

    print("\n========== Configurable fusion CV summary ==========")
    keys = sorted({key for metrics in metrics_per_fold for key in metrics})
    for key in keys:
        values = [metrics[key] for metrics in metrics_per_fold if key in metrics]
        numeric_values = [
            value
            for value in values
            if isinstance(value, Number)
            and not (isinstance(value, float) and math.isnan(value))
        ]
        if numeric_values:
            mean = sum(numeric_values) / len(numeric_values)
            std = (
                sum((value - mean) ** 2 for value in numeric_values)
                / len(numeric_values)
            ) ** 0.5
            print(f"{key}: mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()



"""Train configurable MRI and tabular branches for later decision fusion."""

import argparse
import math
from numbers import Number

from monai.networks.nets import DenseNet121, resnet18
import pytorch_lightning as pl
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.constants import IMAGING_FEATURES_FILE_NAME, SEED
from mriBreastDuke.dataLoaders import (
    CLINICAL_PREDICTOR_COLUMNS,
    IMAGING_FEATURE_GROUPS,
    SENSITIVE_CLINICAL_PREDICTOR_COLUMNS,
    get_oncotype_clinical_predictors_as_study_df,
    merge_precomputed_imaging_features,
    SUBTRACTION_MODES,
    SUBTRACTION_NONE,
    get_input_channels,
)
from mriBreastDuke.n_fold_cv_run import run_5fold_cv


MRI_MODELS = ("fcn", "densenet121", "resnet18")
FEATURE_GROUPS = ("clinical", *IMAGING_FEATURE_GROUPS)
FEATURE_MODELS = ("xgboost", "mlp")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train patient-grouped MRI and precomputed imaging-feature models "
            "and save fold artifacts for post-training fusion validation."
        )
    )
    parser.add_argument("--mri_model", choices=MRI_MODELS, default="densenet121")
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
        choices=FEATURE_MODELS,
        default="xgboost",
    )
    parser.add_argument(
        "--imaging_features_file",
        default=IMAGING_FEATURES_FILE_NAME,
        help="Path to Imaging_Features.xlsx or its CSV export.",
    )
    parser.add_argument("--imaging_patient_id_column", default="Patient ID")
    parser.add_argument("--allow_missing_imaging_features", action="store_true")
    parser.add_argument("--include_sensitive", action="store_true")

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--positive_boost", type=float, default=1.0)
    parser.add_argument("--sensitivity_lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
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


def make_mri_network(model_name, input_channels, num_classes):
    if model_name == "fcn":
        return Simple3DFCN(
            num_classes=num_classes,
            in_channels=input_channels,
        )
    if model_name == "densenet121":
        return DenseNet121(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=num_classes,
        )
    return resnet18(
        spatial_dims=3,
        n_input_channels=input_channels,
        num_classes=num_classes,
    )


def _parse_hidden_layers(value):
    layers = tuple(
        int(width.strip())
        for width in value.split(",")
        if width.strip()
    )
    if not layers or any(width <= 0 for width in layers):
        raise ValueError(
            "--mlp_hidden_layers must contain comma-separated positive integers."
        )
    return layers


def make_feature_model(args, num_classes):
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

    return MLPClassifier(
        hidden_layer_sizes=_parse_hidden_layers(args.mlp_hidden_layers),
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


def prepare_studies_and_features(args):
    selected_groups = tuple(dict.fromkeys(args.feature_groups))
    imaging_groups = tuple(
        group for group in selected_groups if group in IMAGING_FEATURE_GROUPS
    )
    studies = get_oncotype_clinical_predictors_as_study_df(
        isBinary=True,
        include_sensitive=args.include_sensitive,
        include_oncotype_score=False,
    )

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

    if imaging_groups:
        studies, imaging_columns = merge_precomputed_imaging_features(
            studies,
            path=args.imaging_features_file,
            selected_groups=imaging_groups,
            patient_id_column="patientId",
            source_patient_id_column=args.imaging_patient_id_column,
            require_complete_match=not args.allow_missing_imaging_features,
        )
        continuous_columns.extend(imaging_columns)

    return studies, continuous_columns, categorical_columns, selected_groups


def build_experiment_name(args, selected_groups):
    """Return the shared checkpoint/log directory name for one grid entry."""
    return (
        f"ImagingFeaturesFusion_{args.mri_model}_{args.subtraction_mode}_"
        f"{'-'.join(selected_groups)}_{args.feature_model}_lr{args.lr:.0e}_"
        f"sens{args.sensitivity_lambda:g}_boost{args.positive_boost:g}_"
        f"bs{args.batch_size}"
    )


def summarize_metrics(metrics_per_fold):
    print("\n========== Imaging-features training CV summary ==========")
    metric_names = sorted(
        {name for fold_metrics in metrics_per_fold for name in fold_metrics}
    )
    for metric_name in metric_names:
        values = [
            fold_metrics[metric_name]
            for fold_metrics in metrics_per_fold
            if metric_name in fold_metrics
        ]
        numeric_values = [
            value
            for value in values
            if isinstance(value, Number)
            and not (isinstance(value, float) and math.isnan(value))
        ]
        if not numeric_values:
            continue
        mean = sum(numeric_values) / len(numeric_values)
        variance = sum(
            (value - mean) ** 2 for value in numeric_values
        ) / len(numeric_values)
        print(f"{metric_name}: mean={mean:.4f}, std={variance ** 0.5:.4f}")


def main():
    args = parse_args()
    studies, continuous, categorical, selected_groups = (
        prepare_studies_and_features(args)
    )
    num_classes = int(studies["label"].nunique())
    input_channels = get_input_channels(args.subtraction_mode)

    def image_model_factory(class_weights):
        network = make_mri_network(
            args.mri_model,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        return NiftiClassifier(
            network,
            num_classes=num_classes,
            lr=args.lr,
            class_weights=class_weights,
            sensitivity_lambda=args.sensitivity_lambda,
        )

    def feature_model_factory():
        return make_feature_model(args, num_classes)

    experiment_name = build_experiment_name(args, selected_groups)
    metrics = run_5fold_cv(
        df=studies,
        model_name=experiment_name,
        make_model=image_model_factory,
        epoch=args.epoch,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_boost=args.positive_boost,
        subtraction_mode=args.subtraction_mode,
        clinical_continuous_columns=continuous,
        clinical_categorical_columns=categorical,
        group_column="patientId",
        tabular_model_factory=feature_model_factory,
        tabular_model_name=args.feature_model,
    )
    summarize_metrics(metrics)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

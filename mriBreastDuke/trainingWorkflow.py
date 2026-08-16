from monai.networks.nets import DenseNet121, resnet18
import pytorch_lightning as pl
import argparse
from numbers import Number
import math

from mriBreastDuke.dataLoaders import get_oncotype_score_for_series_as_studyId_and_label_df
from mriBreastDuke.constants import SEED
from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.n_fold_cv_run import run_5fold_cv
from mriBreastDuke.dataLoaders.subtraction import (
    SUBTRACTION_MODES,
    SUBTRACTION_NONE,
    get_input_channels,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="0=FCN, 1=DenseNet121, 2=ResNet18",
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--is_binary_classification", type=bool, default=False)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sensitivity_lambda", type=float, default=0.3)
    parser.add_argument("--positive_boost", type=float, default=1.0)
    parser.add_argument(
        "--subtraction_mode",
        choices=SUBTRACTION_MODES,
        default=SUBTRACTION_NONE,
        help=(
            "MRI input transformation: none keeps the original series; "
            "post_minus_pre subtracts series[0] from every later series; "
            "consecutive subtracts each series from the following series."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = get_oncotype_score_for_series_as_studyId_and_label_df(
        args.is_binary_classification
    )
    num_classes = len(set(df.label))
    input_channels = get_input_channels(args.subtraction_mode)
    subtraction_suffix = (
        "" if args.subtraction_mode == SUBTRACTION_NONE
        else f"_sub_{args.subtraction_mode}"
    )

    run_name_suffix = (
        f"lr_{args.lr:.0e}"
        f"_sens_{args.sensitivity_lambda}"
        f"_posboost_{args.positive_boost}"
        f"_bs_{args.batch_size}"
        f"{subtraction_suffix}"
    )

    models = [
        (
            f"FCN_{run_name_suffix}",
            lambda class_weights=None: NiftiClassifier(
                Simple3DFCN(
                    num_classes=num_classes,
                ),
                num_classes,
                lr=args.lr,
                class_weights=class_weights,
                sensitivity_lambda=args.sensitivity_lambda,
            ),
        ),
        (
            f"DenseNet_{run_name_suffix}",
            lambda class_weights=None: NiftiClassifier(
                DenseNet121(
                    spatial_dims=3,
                    in_channels=input_channels,
                    out_channels=num_classes,
                ),
                num_classes,
                lr=args.lr,
                class_weights=class_weights,
                sensitivity_lambda=args.sensitivity_lambda,
            ),
        ),
        (
            f"ResNet18_{run_name_suffix}",
            lambda class_weights=None: NiftiClassifier(
                resnet18(
                    spatial_dims=3,
                    n_input_channels=input_channels,
                    num_classes=num_classes,
                ),
                num_classes,
                lr=args.lr,
                class_weights=class_weights,
                sensitivity_lambda=args.sensitivity_lambda,
            ),
        ),
    ]

    model_name, make_model = models[args.model]

    metrics_per_fold = run_5fold_cv(
        df=df,
        model_name=model_name,
        make_model=make_model,
        epoch=args.epoch,
        num_folds=5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_boost=args.positive_boost,
        subtraction_mode=args.subtraction_mode,
    )

    print("\n========== CV Summary ==========")
    keys = sorted({k for m in metrics_per_fold for k in m.keys()})

    for k in keys:
        raw_vals = [m[k] for m in metrics_per_fold if k in m]
        vals = [
            v for v in raw_vals
            if isinstance(v, Number) and not (isinstance(v, float) and math.isnan(v))
        ]

        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"{k}: mean={mean:.4f}, std={std:.4f}")
        else:
            print(f"{k}: non-numeric, skipped")


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

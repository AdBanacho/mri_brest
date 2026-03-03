from monai.networks.nets import DenseNet121
import pytorch_lightning as pl
import argparse

from mriBreastDuke.dataLoaders import get_oncotype_score_for_series_as_studyId_and_label_df
from mriBreastDuke.constants import SEED
from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.n_fold_cv_run import run_5fold_cv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=int, default=0,
                        help="0=FCN, 1=DenseNet, 2=ViT3D")
    parser.add_argument("--epoch", type=int, default=50)

    return parser.parse_args()
    
def main():
    df = get_oncotype_score_for_series_as_studyId_and_label_df()
    args = parse_args()
    num_classes = len(set(df.label))

    models = [
        ("FCN", lambda: NiftiClassifier(Simple3DFCN(num_classes=num_classes), num_classes)),
        ("DenseNet", lambda: NiftiClassifier(DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_classes), num_classes)),
    ]

    model_name, make_model = models[args.model]
    metrics_per_fold = run_5fold_cv(df, model_name, make_model, args.epoch, num_folds=5)

    # Optional: simple aggregation printout
    print("\n========== CV Summary ==========")
    # pick metrics you care about (example keys)
    keys = sorted({k for m in metrics_per_fold for k in m.keys()})
    for k in keys:
        vals = [m[k] for m in metrics_per_fold if k in m]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"{k}: mean={mean:.4f}, std={std:.4f}")


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

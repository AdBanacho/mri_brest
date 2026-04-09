from pathlib import Path
import argparse
from numbers import Number
import math
import re

from monai.networks.nets import DenseNet121
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl

from mriBreastDuke.constants import CHECKPOINTS, NIFTI_PATH, SEED
from mriBreastDuke.classificators import NiftiClassifier, Simple3DFCN
from mriBreastDuke.dataLoaders import (
    NiftiDataModule,
    get_oncotype_score_for_series_as_studyId_and_label_df,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load best checkpoints from each CV fold and run validation."
    )
    parser.add_argument("--model", type=int, default=0, help="0=FCN, 1=DenseNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=CHECKPOINTS,
        help="Root directory that contains model fold checkpoints.",
    )
    return parser.parse_args()


BEST_CKPT_PATTERN = re.compile(r"best-epoch=(?P<epoch>\d+)-val_loss=(?P<val_loss>[0-9]*\.?[0-9]+)\.ckpt$")


def get_best_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.glob("best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No best-*.ckpt file found in: {checkpoint_dir}")

    parsed = []
    for path in candidates:
        match = BEST_CKPT_PATTERN.match(path.name)
        if not match:
            continue
        val_loss = float(match.group("val_loss"))
        epoch = int(match.group("epoch"))
        parsed.append((val_loss, -epoch, path))

    if parsed:
        parsed.sort(key=lambda x: (x[0], x[1]))
        return parsed[0][2]

    return min(candidates, key=lambda p: p.stat().st_mtime)


def resolve_fold_checkpoint_dir(checkpoint_root: str, model_name: str, fold: int) -> Path:
    root = Path(checkpoint_root)

    candidates = [
        root / model_name / f"fold_{fold}" / "checkpoints",
        root / f"fold_{fold}" / "checkpoints",
        root / f"fold_{fold}",
    ]

    if root.name == "checkpoints" and root.parent.name.startswith("fold_"):
        model_dir = root.parent.parent
        candidates.append(model_dir / f"fold_{fold}" / "checkpoints")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def run_saved_checkpoint_validation(df, model_name, make_model, num_folds, batch_size, num_workers, checkpoint_root):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    y = df["label"].values
    fold_metrics = []

    for fold, (_, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n========== Fold {fold}/{num_folds} ==========")

        val_df = df.iloc[val_idx].reset_index(drop=True)

        datamodule = NiftiDataModule(
            train_df=val_df,  # required by DataModule, but unused by validate()
            val_df=val_df,
            target_size=(256, 256, 64),
            image_root=NIFTI_PATH,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        checkpoint_dir = resolve_fold_checkpoint_dir(checkpoint_root, model_name, fold)
        checkpoint_path = get_best_checkpoint(checkpoint_dir)

        trainer = pl.Trainer(logger=False, enable_progress_bar=True)

        metrics = trainer.validate(
            model=make_model(),
            datamodule=datamodule,
            ckpt_path=str(checkpoint_path),
            verbose=False,
        )[0]

        serialized_metrics = {
            k: float(v) if isinstance(v, Number) else v
            for k, v in metrics.items()
        }
        serialized_metrics["checkpoint_path"] = str(checkpoint_path)
        fold_metrics.append(serialized_metrics)

        print(f"Checkpoint: {checkpoint_path}")
        for k, v in serialized_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return fold_metrics


def print_summary(metrics_per_fold):
    print("\n========== Validation Summary ==========")
    keys = sorted({k for fold_metrics in metrics_per_fold for k in fold_metrics.keys()})

    for k in keys:
        values = [m[k] for m in metrics_per_fold if k in m]
        numeric_values = [
            v for v in values
            if isinstance(v, Number) and not (isinstance(v, float) and math.isnan(v))
        ]

        if numeric_values:
            mean = sum(numeric_values) / len(numeric_values)
            std = (sum((v - mean) ** 2 for v in numeric_values) / len(numeric_values)) ** 0.5
            print(f"{k}: mean={mean:.4f}, std={std:.4f}")


def main():
    args = parse_args()

    df = get_oncotype_score_for_series_as_studyId_and_label_df()
    num_classes = len(set(df.label))

    models = [
        ("FCN", lambda: NiftiClassifier(Simple3DFCN(num_classes=num_classes), num_classes)),
        (
            "DenseNet",
            lambda: NiftiClassifier(
                DenseNet121(spatial_dims=3, in_channels=5, out_channels=num_classes),
                num_classes,
            ),
        ),
    ]

    model_name, make_model = models[args.model]

    metrics_per_fold = run_saved_checkpoint_validation(
        df=df,
        model_name=model_name,
        make_model=make_model,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_root=args.checkpoint_root,
    )

    print_summary(metrics_per_fold)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

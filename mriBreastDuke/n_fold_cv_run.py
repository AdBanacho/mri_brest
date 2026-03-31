from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch

from mriBreastDuke.classificators import DebugBatchShapeCallback
from mriBreastDuke.constants import SEED, LIGHTING_LOGS, NIFTI_PATH, CHECKPOINTS
from mriBreastDuke.dataLoaders import NiftiDataModule


def run_5fold_cv(df, model_name, make_model, epoch, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    y = df["label"].values
    metrics_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n========== Fold {fold}/{num_folds} ==========")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        datamodule = NiftiDataModule(
            train_df=train_df,
            val_df=val_df,
            target_size=(256, 256, 64),
            image_root=NIFTI_PATH,
            batch_size=8,
            num_workers=2,
        )

        model = make_model()

        logger = TensorBoardLogger(
            save_dir=LIGHTING_LOGS,
            name=model_name,
            version=f"fold_{fold}",
        )

        # Directory for this fold's checkpoints
        ckpt_dir = Path(CHECKPOINTS) / model_name / f"fold_{fold}" / "checkpoints"

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",   # change if your validation metric has another name
            mode="min",           # "min" for loss, "max" for metrics like val_acc
            save_top_k=1,         # keep only the best epoch
            save_last=True,       # optional: also keep the last epoch
            verbose=True,
        )

        trainer = pl.Trainer(
            max_epochs=epoch,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[
                DebugBatchShapeCallback(),
                checkpoint_callback,
            ],
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model=model, datamodule=datamodule)

        # Metrics from last epoch (already present)
        fold_metrics = {
            k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item")
        }

        # Evaluate BEST checkpoint
        best_metrics = trainer.validate(
            model=make_model(),
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path,
            verbose=False
        )[0]

        # Store both
        fold_metrics["best_model_path"] = checkpoint_callback.best_model_path
        fold_metrics["best_val_loss"] = float(checkpoint_callback.best_model_score)

        # Add best checkpoint metrics
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                fold_metrics[f"best_{k}"] = float(v)

        print(f"\nFold {fold} metrics:")
        for k, v in fold_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

    return metrics_per_fold
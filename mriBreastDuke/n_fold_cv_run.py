from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torch

from mriBreastDuke.classificators import DebugBatchShapeCallback
from mriBreastDuke.constants import SEED, LIGHTING_LOGS, NIFTI_PATH
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

        # IMPORTANT: fresh model + fresh trainer per fold
        model = make_model()

        logger = TensorBoardLogger(
            save_dir=LIGHTING_LOGS,
            name=f"{model_name}/fold_{fold}",
        )

        trainer = pl.Trainer(
            max_epochs=epoch,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[DebugBatchShapeCallback()],
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model=model, datamodule=datamodule)

        fold_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item")}
        metrics_per_fold.append(fold_metrics)

        print(f"\nFold {fold} metrics:")
        for k, v in fold_metrics.items():
            print(f"  {k}: {v:.4f}")

    return metrics_per_fold
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mriBreastDuke.classificators import DebugBatchShapeCallback
from mriBreastDuke.constants import SEED, LIGHTING_LOGS, NIFTI_PATH, CHECKPOINTS
from mriBreastDuke.dataLoaders import NiftiDataModule


def run_5fold_cv(df, model_name, make_model, epoch, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    y = df["label"].values
    metrics_per_fold = []
    histories_per_fold = []

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

        class_weights = _compute_balanced_class_weights(train_df["label"].values)
        model = make_model(class_weights=class_weights)

        logger = TensorBoardLogger(
            save_dir=LIGHTING_LOGS,
            name=model_name,
            version=f"fold_{fold}",
        )

        # Directory for this fold's checkpoints
        ckpt_dir = Path(CHECKPOINTS) / model_name / f"fold_{fold}" / "checkpoints"

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_sensitivity:.4f}",
            monitor="val_sensitivity",
            mode="max",
            save_top_k=1,         # keep only the best epoch
            save_last=True,       # optional: also keep the last epoch
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor="val_sensitivity",
            mode="max",
            patience=8,
            min_delta=1e-4,
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
                early_stopping,
            ],
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model=model, datamodule=datamodule)

        fold_history = {
            "fold": fold,

            "train_loss": list(model.train_loss_history),
            "train_ce_loss": list(model.train_ce_loss_history),
            "train_sensitivity_loss": list(model.train_sensitivity_loss_history),
            "train_soft_sensitivity": list(model.train_soft_sensitivity_history),

            "val_loss": list(model.val_loss_history),
            "val_ce_loss": list(model.val_ce_loss_history),
            "val_sensitivity_loss": list(model.val_sensitivity_loss_history),
            "val_soft_sensitivity": list(model.val_soft_sensitivity_history),
        }

        histories_per_fold.append(fold_history)

        # Metrics from last epoch (already present)
        fold_metrics = {
            k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "item")
        }

        # Evaluate BEST checkpoint
        best_metrics = trainer.validate(
            model=make_model(class_weights=class_weights),
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path,
            verbose=False
        )[0]

        # Store both
        fold_metrics["best_model_path"] = checkpoint_callback.best_model_path
        fold_metrics["best_val_sensitivity_checkpoint_score"] = float(checkpoint_callback.best_model_score)

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

        metrics_per_fold.append(fold_metrics)

    summary_dir = Path(LIGHTING_LOGS) / model_name / "cross_validation_summary"
    summary_writer = SummaryWriter(log_dir=str(summary_dir))

    _plot_cv_metric_with_folds(
        histories_per_fold,
        metric_key="train_loss",
        val_metric_key="val_loss",
        title="Total Loss Across Folds",
        ylabel="Loss",
        writer=summary_writer,
        tag="cv/total_loss",
    )

    _plot_cv_metric_with_folds(
        histories_per_fold,
        metric_key="train_ce_loss",
        val_metric_key="val_ce_loss",
        title="Cross-Entropy Loss Across Folds",
        ylabel="CE Loss",
        writer=summary_writer,
        tag="cv/ce_loss",
    )

    _plot_cv_metric_with_folds(
        histories_per_fold,
        metric_key="train_sensitivity_loss",
        val_metric_key="val_sensitivity_loss",
        title="Sensitivity Loss Across Folds",
        ylabel="Sensitivity Loss",
        writer=summary_writer,
        tag="cv/sensitivity_loss",
    )

    _plot_cv_metric_with_folds(
        histories_per_fold,
        metric_key="train_soft_sensitivity",
        val_metric_key="val_soft_sensitivity",
        title="Soft Sensitivity Across Folds",
        ylabel="Soft Sensitivity",
        writer=summary_writer,
        tag="cv/soft_sensitivity",
        ylim=(0.0, 1.0),
    )

    summary_writer.flush()
    summary_writer.close()

    return metrics_per_fold

def _plot_cv_metric_with_folds(
    histories_per_fold,
    metric_key,
    val_metric_key,
    title,
    ylabel,
    writer,
    tag,
    ylim=None,
):
    """
    Plots every fold separately and also plots mean ± std across folds.
    Adds the figure directly to TensorBoard instead of saving PNG files.

    Handles early stopping by trimming all folds to the shortest available history.
    """
    train_histories = [
        h[metric_key]
        for h in histories_per_fold
        if len(h.get(metric_key, [])) > 0
    ]

    val_histories = [
        h[val_metric_key]
        for h in histories_per_fold
        if len(h.get(val_metric_key, [])) > 0
    ]

    if len(train_histories) == 0 and len(val_histories) == 0:
        print(f"[CV PLOT] No data for {metric_key} / {val_metric_key}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    if len(train_histories) > 0:
        min_train_len = min(len(h) for h in train_histories)
        train_arr = np.array([h[:min_train_len] for h in train_histories])

        train_epochs = np.arange(1, min_train_len + 1)
        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)

        for i, h in enumerate(train_arr, start=1):
            ax.plot(
                train_epochs,
                h,
                alpha=0.25,
                label=f"fold {i} train",
            )

        ax.plot(
            train_epochs,
            train_mean,
            linewidth=2.5,
            label="train mean",
        )

        ax.fill_between(
            train_epochs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            label="train ± std",
        )

    if len(val_histories) > 0:
        min_val_len = min(len(h) for h in val_histories)
        val_arr = np.array([h[:min_val_len] for h in val_histories])

        val_epochs = np.arange(1, min_val_len + 1)
        val_mean = val_arr.mean(axis=0)
        val_std = val_arr.std(axis=0)

        for i, h in enumerate(val_arr, start=1):
            ax.plot(
                val_epochs,
                h,
                linestyle="--",
                alpha=0.25,
                label=f"fold {i} val",
            )

        ax.plot(
            val_epochs,
            val_mean,
            linestyle="--",
            linewidth=2.5,
            label="val mean",
        )

        ax.fill_between(
            val_epochs,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            label="val ± std",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()

    writer.add_figure(tag, fig, global_step=0)
    plt.close(fig)

    print(f"[CV TENSORBOARD] Added figure: {tag}")

def _compute_balanced_class_weights(labels):
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor)
    total = class_counts.sum().float()
    num_classes = class_counts.numel()
    weights = total / (num_classes * class_counts.float().clamp_min(1.0))
    return weights

from pathlib import Path
import matplotlib.pyplot as plt

import joblib
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mriBreastDuke.constants import SEED, LIGHTING_LOGS, NIFTI_PATH, CHECKPOINTS_PATH
from mriBreastDuke.dataLoaders import ClinicalFeaturePreprocessor, NiftiDataModule, SUBTRACTION_NONE
from mriBreastDuke.classificators import (
    aligned_predict_proba,
    fuse_probabilities,
    probability_metrics,
    save_fusion_predictions,
    DebugBatchShapeCallback
)

def _resolve_output_dir(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _predict_image_probabilities(model, dataloader, num_classes):
    probabilities = []
    labels = []
    model.eval()
    device = model.device
    with torch.no_grad():
        for images, batch_labels in dataloader:
            logits = model(images.to(device))
            if num_classes == 2 and (logits.ndim == 1 or logits.shape[1] == 1):
                positive = torch.sigmoid(logits.view(-1))
                batch_probabilities = torch.stack((1.0 - positive, positive), dim=1)
            else:
                batch_probabilities = torch.softmax(logits, dim=1)
            probabilities.append(batch_probabilities.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(probabilities), np.concatenate(labels)

def run_5fold_cv(
    df,
    model_name,
    make_model,
    epoch,
    num_folds=5,
    batch_size=8,
    num_workers=2,
    positive_boost=1.0,
    subtraction_mode=SUBTRACTION_NONE,
    clinical_continuous_columns=None,
    clinical_categorical_columns=None,
    group_column=None,
    tabular_model_factory=None,
    fusion_alpha=0.5,
):
    use_tabular_features = bool(clinical_continuous_columns or clinical_categorical_columns)
    if use_tabular_features != (tabular_model_factory is not None):
        raise ValueError(
            "Tabular columns and tabular_model_factory must be provided together."
        )
    if not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be between 0 and 1.")
    if group_column is not None:
        if group_column not in df.columns:
            raise ValueError(f"group_column '{group_column}' is not present in the DataFrame.")
        skf = StratifiedGroupKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=SEED,
        )
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)

    logs_root = _resolve_output_dir(LIGHTING_LOGS)
    checkpoints_root = _resolve_output_dir(CHECKPOINTS_PATH)

    print(f"[LOGS] TensorBoard root: {logs_root}", flush=True)
    print(f"[CKPT] Checkpoint root: {checkpoints_root}", flush=True)
    print(f"[INPUT] Subtraction mode: {subtraction_mode}", flush=True)

    y = df["label"].values
    groups = df[group_column].values if group_column is not None else None
    metrics_per_fold = []
    histories_per_fold = []

    split_iterator = skf.split(df, y, groups) if groups is not None else skf.split(df, y)
    for fold, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        print(f"\n========== Fold {fold}/{num_folds} ==========")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        clinical_preprocessor = None
        tabular_model = None
        tabular_validation_probabilities = None
        if use_tabular_features:
            clinical_preprocessor = ClinicalFeaturePreprocessor(
                continuous_columns=clinical_continuous_columns or (),
                categorical_columns=clinical_categorical_columns or (),
            )
            train_tabular = clinical_preprocessor.fit_transform(train_df)
            validation_tabular = clinical_preprocessor.transform(val_df)
            tabular_model = tabular_model_factory()
            tabular_labels = train_df["label"].values
            tabular_sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=tabular_labels,
            )
            if int(df["label"].nunique()) == 2 and positive_boost != 1.0:
                tabular_sample_weights = tabular_sample_weights.copy()
                tabular_sample_weights[tabular_labels == 1] *= positive_boost
            tabular_model.fit(
                train_tabular,
                tabular_labels,
                sample_weight=tabular_sample_weights,
            )
            tabular_validation_probabilities = aligned_predict_proba(
                tabular_model,
                validation_tabular,
                num_classes=int(df["label"].nunique()),
            )
            print(
                f"[Fold {fold}] XGBoost input dimension: "
                f"{clinical_preprocessor.output_dimension}",
                flush=True,
            )

        datamodule = NiftiDataModule(
            train_df=train_df,
            val_df=val_df,
            target_size=(256, 256, 64),
            image_root=NIFTI_PATH,
            batch_size=batch_size,
            num_workers=num_workers,
            subtraction_mode=subtraction_mode,
        )

        class_weights = _compute_balanced_class_weights(
            train_df["label"].values,
            positive_boost=positive_boost,
        )
        model = make_model(class_weights=class_weights)

        fold_version = f"fold_{fold}"

        logger = TensorBoardLogger(
            save_dir=str(logs_root),
            name=model_name,
            version=fold_version,
            default_hp_metric=False,
        )

        # Force creation early, so the folder appears even before first scalar is written.
        Path(logger.log_dir).mkdir(parents=True, exist_ok=True)

        print(f"[Fold {fold}] TensorBoard log dir: {logger.log_dir}", flush=True)

        # Directory for this fold's checkpoints
        ckpt_dir = checkpoints_root / model_name / fold_version / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if clinical_preprocessor is not None:
            preprocessor_path = ckpt_dir / "clinical_preprocessor.joblib"
            joblib.dump(clinical_preprocessor, preprocessor_path)
            feature_names_path = ckpt_dir / "clinical_feature_names.txt"
            feature_names_path.write_text(
                "\n".join(clinical_preprocessor.get_feature_names_out()) + "\n",
                encoding="utf-8",
            )
            print(
                f"[Fold {fold}] Saved tabular preprocessor: {preprocessor_path}",
                flush=True,
            )
            xgboost_path = ckpt_dir / "xgboost_model.joblib"
            joblib.dump(tabular_model, xgboost_path)
            print(f"[Fold {fold}] Saved XGBoost model: {xgboost_path}", flush=True)

        print(f"[Fold {fold}] Checkpoint dir: {ckpt_dir}", flush=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_sensitivity:.4f}-{val_auc_roc:.4f}",
            monitor="val_sensitivity",
            mode="max",
            save_top_k=1,
            save_last=True,
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
            default_root_dir=str(logs_root / model_name / fold_version),
            callbacks=[
                DebugBatchShapeCallback(),
                checkpoint_callback,
                #early_stopping,
            ],
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model=model, datamodule=datamodule)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.flush()

        fusion_metrics = {}
        if tabular_model is not None:
            if checkpoint_callback.best_model_path:
                checkpoint = torch.load(
                    checkpoint_callback.best_model_path,
                    map_location=model.device,
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["state_dict"])

            image_probabilities, validation_labels = _predict_image_probabilities(
                model,
                datamodule.val_dataloader(),
                num_classes=int(df["label"].nunique()),
            )
            fused_probabilities = fuse_probabilities(
                image_probabilities,
                tabular_validation_probabilities,
                alpha=fusion_alpha,
            )
            fusion_metrics.update(
                probability_metrics(
                    validation_labels,
                    image_probabilities,
                    prefix="image",
                )
            )
            fusion_metrics.update(
                probability_metrics(
                    validation_labels,
                    tabular_validation_probabilities,
                    prefix="xgboost",
                )
            )
            fusion_metrics.update(
                probability_metrics(
                    validation_labels,
                    fused_probabilities,
                    prefix="fusion",
                )
            )
            save_fusion_predictions(
                val_df,
                validation_labels,
                image_probabilities,
                tabular_validation_probabilities,
                fused_probabilities,
                ckpt_dir / "fusion_validation_predictions.csv",
            )
            print(
                f"[Fold {fold}] Decision fusion alpha (MRI weight): {fusion_alpha}",
                flush=True,
            )

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
        fold_metrics.update(fusion_metrics)

        # Evaluate BEST checkpoint
        # best_metrics = trainer.validate(
        #     model=make_model(class_weights=class_weights),
        #     datamodule=datamodule,
        #     ckpt_path=checkpoint_callback.best_model_path,
        #     verbose=False
        # )[0]

        # Final flush/save for this fold.
        if logger is not None:
            if hasattr(logger, "save"):
                logger.save()

            if hasattr(logger, "experiment"):
                logger.experiment.flush()
                logger.experiment.close()

            if hasattr(logger, "finalize"):
                logger.finalize("success")

        # Store both
        fold_metrics["best_model_path"] = checkpoint_callback.best_model_path
        fold_metrics["best_val_sensitivity_checkpoint_score"] = float(checkpoint_callback.best_model_score)

        # Add best checkpoint metrics
        # for k, v in best_metrics.items():
        #     if isinstance(v, (int, float)):
        #         fold_metrics[f"best_{k}"] = float(v)

        print(f"\nFold {fold} metrics:")
        for k, v in fold_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

        metrics_per_fold.append(fold_metrics)

    summary_dir = logs_root / model_name / "cross_validation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=str(summary_dir))
    print(f"[CV SUMMARY] TensorBoard log dir: {summary_dir}", flush=True)

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

def _compute_balanced_class_weights(labels, positive_boost=1.0):
    label_tensor = torch.as_tensor(labels, dtype=torch.long)

    class_counts = torch.bincount(label_tensor)
    total = class_counts.sum().float()
    num_classes = class_counts.numel()

    weights = total / (num_classes * class_counts.float().clamp_min(1.0))

    if num_classes == 2:
        weights[1] *= positive_boost

    print(f"[CLASS WEIGHTS] counts={class_counts.tolist()} weights={weights.tolist()}", flush=True)

    return weights

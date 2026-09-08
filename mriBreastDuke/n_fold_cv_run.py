import gc
from pathlib import Path
import matplotlib.pyplot as plt

import joblib
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from mriBreastDuke.constants import SEED, LIGHTING_LOGS, NIFTI_PATH, CHECKPOINTS_PATH
from mriBreastDuke.dataLoaders import (
    ClinicalFeaturePreprocessor,
    NiftiDataModule,
    SUBTRACTION_NONE,
    save_lasso_feature_importance_chart,
)
from mriBreastDuke.classificators import DebugBatchShapeCallback

def _resolve_output_dir(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _scalar_metrics(metrics):
    """Detach scalar trainer metrics so they cannot retain device memory."""
    detached = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            detached[name] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, np.number)):
            detached[name] = float(value)
    return detached


class _BestValidationMetrics(pl.Callback):
    """Keep the metrics from the epoch selected by ModelCheckpoint.

    This avoids running a second full validation loop after ``Trainer.fit``.
    The callback deliberately ignores Lightning's sanity-validation pass.
    """

    def __init__(self, monitor, mode="max"):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_metrics = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = _scalar_metrics(trainer.callback_metrics)
        current = metrics.get(self.monitor)
        if current is None or not np.isfinite(current):
            return

        is_better = self.best_score is None or (
            current > self.best_score
            if self.mode == "max"
            else current < self.best_score
        )
        if is_better:
            self.best_score = current
            self.best_metrics = metrics


def _validate_cv_inputs(
    df,
    num_folds,
    group_column,
    continuous_columns,
    categorical_columns,
    tabular_model_factory,
):
    """Validate labels, groups, and merged tabular columns before training."""
    if "label" not in df.columns:
        raise ValueError("The cross-validation DataFrame must contain 'label'.")
    if len(df) == 0:
        raise ValueError("The cross-validation DataFrame is empty.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if df["label"].isna().any():
        raise ValueError("The label column contains missing values.")

    labels = df["label"].to_numpy()
    integer_labels = labels.astype(np.int64)
    if not np.array_equal(labels, integer_labels):
        raise ValueError("Labels must be integers encoded from 0 to C-1.")
    unique_labels = np.unique(integer_labels)
    expected_labels = np.arange(len(unique_labels))
    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            "Labels must be contiguous integers encoded from 0 to C-1; "
            f"received {unique_labels.tolist()}."
        )
    if len(unique_labels) < 2:
        raise ValueError("Cross-validation requires at least two classes.")

    continuous_columns = tuple(continuous_columns or ())
    categorical_columns = tuple(categorical_columns or ())
    all_tabular_columns = (*continuous_columns, *categorical_columns)
    if len(set(all_tabular_columns)) != len(all_tabular_columns):
        raise ValueError(
            "Each tabular feature must appear exactly once across continuous "
            "and categorical columns."
        )
    missing_columns = sorted(set(all_tabular_columns).difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing tabular feature columns: {missing_columns[:10]}"
        )
    if "label" in all_tabular_columns:
        raise ValueError("The target label cannot be used as a tabular feature.")

    use_tabular_features = bool(all_tabular_columns)
    if use_tabular_features != (tabular_model_factory is not None):
        raise ValueError(
            "Tabular columns and tabular_model_factory must be provided together."
        )
    if group_column is None:
        class_counts = np.bincount(integer_labels, minlength=len(unique_labels))
        if np.any(class_counts < num_folds):
            raise ValueError(
                "Every class must contain at least num_folds rows for "
                f"StratifiedKFold; counts={class_counts.tolist()}, "
                f"num_folds={num_folds}."
            )
    else:
        if group_column not in df.columns:
            raise ValueError(
                f"group_column '{group_column}' is not present in the DataFrame."
            )
        if df[group_column].isna().any():
            raise ValueError(f"group_column '{group_column}' contains missing values.")

        group_label_counts = df.groupby(group_column, sort=False)["label"].nunique()
        inconsistent = group_label_counts[group_label_counts > 1]
        if not inconsistent.empty:
            examples = ", ".join(map(str, inconsistent.index[:5]))
            raise ValueError(
                f"Each '{group_column}' group must have one label; inconsistent "
                f"groups include: {examples}."
            )

        unique_group_labels = df[[group_column, "label"]].drop_duplicates()
        groups_per_class = (
            unique_group_labels.groupby("label")[group_column]
            .nunique()
            .reindex(expected_labels, fill_value=0)
            .to_numpy()
        )
        if np.any(groups_per_class < num_folds):
            raise ValueError(
                "Every class must contain at least num_folds distinct patient "
                f"groups; counts={groups_per_class.tolist()}, "
                f"num_folds={num_folds}."
            )

    return integer_labels, continuous_columns, categorical_columns


def _validate_fold_separation(
    train_df,
    validation_df,
    group_column,
    expected_classes,
):
    """Fail early if a generated fold loses a class or leaks a patient group."""
    train_classes = np.unique(train_df["label"].to_numpy(dtype=np.int64))
    if not np.array_equal(train_classes, expected_classes):
        raise ValueError(
            "A training fold is missing one or more classes; "
            f"received {train_classes.tolist()}, expected {expected_classes.tolist()}."
        )
    if group_column is not None:
        overlap = set(train_df[group_column]).intersection(validation_df[group_column])
        if overlap:
            examples = ", ".join(map(str, list(overlap)[:5]))
            raise RuntimeError(
                f"Patient-group leakage detected between train and validation: {examples}."
            )


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
    tabular_model_name="xgboost",
    tabular_feature_selector_factory=None,
    tabular_feature_plot_top_n=30,
):
    """Train MRI and optional tabular branches in leakage-safe CV folds.

    Precomputed ``Imaging_Features.xlsx`` columns must be merged into ``df``
    before this function is called. Pass those numeric columns through
    ``clinical_continuous_columns``; despite the legacy parameter name, they
    are treated as generic tabular predictors and are imputed/scaled using the
    training portion of each fold only. An optional tabular feature selector is
    also fitted only on that training portion and receives patient groups for
    leakage-safe inner cross-validation.
    """
    (
        y,
        continuous_columns,
        categorical_columns,
    ) = _validate_cv_inputs(
        df=df,
        num_folds=num_folds,
        group_column=group_column,
        continuous_columns=clinical_continuous_columns,
        categorical_columns=clinical_categorical_columns,
        tabular_model_factory=tabular_model_factory,
    )
    use_tabular_features = bool(continuous_columns or categorical_columns)
    if tabular_feature_selector_factory is not None and not use_tabular_features:
        raise ValueError(
            "A tabular feature selector cannot be used without tabular columns."
        )
    if tabular_feature_plot_top_n < 1:
        raise ValueError("tabular_feature_plot_top_n must be at least 1.")
    num_classes = len(np.unique(y))
    tabular_model_name = str(tabular_model_name).strip()
    if use_tabular_features and not tabular_model_name:
        raise ValueError("tabular_model_name cannot be empty.")

    if group_column is not None:
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

    groups = df[group_column].values if group_column is not None else None
    metrics_per_fold = []
    histories_per_fold = []
    feature_selection_reports = []

    split_iterator = skf.split(df, y, groups) if groups is not None else skf.split(df, y)
    for fold, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        print(f"\n========== Fold {fold}/{num_folds} ==========")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        _validate_fold_separation(
            train_df,
            val_df,
            group_column=group_column,
            expected_classes=np.arange(num_classes),
        )

        tabular_preprocessor = None
        tabular_model = None
        tabular_feature_selector = None
        if use_tabular_features:
            tabular_preprocessor = ClinicalFeaturePreprocessor(
                continuous_columns=continuous_columns,
                categorical_columns=categorical_columns,
            )
            train_tabular = tabular_preprocessor.fit_transform(train_df)
            tabular_model = tabular_model_factory()
            tabular_labels = train_df["label"].values
            tabular_sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=tabular_labels,
            )
            if num_classes == 2 and positive_boost != 1.0:
                tabular_sample_weights = tabular_sample_weights.copy()
                tabular_sample_weights[tabular_labels == 1] *= positive_boost
            preprocessed_dimension = train_tabular.shape[1]
            if tabular_feature_selector_factory is not None:
                tabular_feature_selector = tabular_feature_selector_factory()
                if tabular_feature_selector is None:
                    raise ValueError(
                        "tabular_feature_selector_factory returned None."
                    )
                selector_groups = (
                    train_df[group_column].to_numpy()
                    if group_column is not None
                    else None
                )
                train_tabular = tabular_feature_selector.fit_transform(
                    train_tabular,
                    tabular_labels,
                    sample_weight=tabular_sample_weights,
                    groups=selector_groups,
                )
            tabular_model.fit(
                train_tabular,
                tabular_labels,
                sample_weight=tabular_sample_weights,
            )
            print(
                f"[Fold {fold}] {tabular_model_name} input dimension: "
                f"{train_tabular.shape[1]} "
                f"(preprocessed={preprocessed_dimension})",
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

        if tabular_preprocessor is not None:
            preprocessor_path = ckpt_dir / "tabular_preprocessor.joblib"
            joblib.dump(tabular_preprocessor, preprocessor_path)
            feature_names_path = ckpt_dir / "tabular_feature_names.txt"
            feature_names_path.write_text(
                "\n".join(tabular_preprocessor.get_feature_names_out()) + "\n",
                encoding="utf-8",
            )
            print(
                f"[Fold {fold}] Saved tabular preprocessor: {preprocessor_path}",
                flush=True,
            )
            if tabular_feature_selector is not None:
                selector_path = ckpt_dir / "tabular_feature_selector.joblib"
                joblib.dump(tabular_feature_selector, selector_path)
                all_feature_names = tabular_preprocessor.get_feature_names_out()
                selected_feature_names = (
                    tabular_feature_selector.get_feature_names_out(all_feature_names)
                )
                selected_names_path = (
                    ckpt_dir / "tabular_selected_feature_names.txt"
                )
                selected_names_path.write_text(
                    "\n".join(selected_feature_names) + "\n",
                    encoding="utf-8",
                )
                selection_report = tabular_feature_selector.selection_report(
                    all_feature_names
                )
                selection_report.insert(0, "fold", fold)
                selection_report_path = (
                    ckpt_dir / "lasso_feature_selection.csv"
                )
                selection_report.to_csv(selection_report_path, index=False)
                selection_chart_path = (
                    ckpt_dir / "lasso_feature_importance.png"
                )
                save_lasso_feature_importance_chart(
                    selection_report,
                    selection_chart_path,
                    title=f"Fold {fold} selected LASSO feature importance",
                    top_n=tabular_feature_plot_top_n,
                )
                feature_selection_reports.append(selection_report)
                selected_by_group = (
                    selection_report.loc[selection_report["selected"]]
                    .groupby("feature_group")["feature_name"]
                    .count()
                    .to_dict()
                )
                print(
                    f"[Fold {fold}] LASSO selected "
                    f"{len(selected_feature_names)}/{len(all_feature_names)} "
                    f"features: {selected_by_group}",
                    flush=True,
                )
                print(
                    f"[Fold {fold}] Saved LASSO report: "
                    f"{selection_report_path}",
                    flush=True,
                )
                print(
                    f"[Fold {fold}] Saved LASSO chart: "
                    f"{selection_chart_path}",
                    flush=True,
                )
            tabular_model_path = ckpt_dir / f"{tabular_model_name}_model.joblib"
            joblib.dump(tabular_model, tabular_model_path)
            print(
                f"[Fold {fold}] Saved {tabular_model_name} model: "
                f"{tabular_model_path}",
                flush=True,
            )
            # The saved tabular artifacts are consumed by the standalone
            # validation workflow; MRI training does not need to retain them.
            tabular_preprocessor = None
            tabular_model = None
            tabular_feature_selector = None
            del train_tabular, tabular_labels, tabular_sample_weights
            gc.collect()

        print(f"[Fold {fold}] Checkpoint dir: {ckpt_dir}", flush=True)

        best_metrics_callback = _BestValidationMetrics(
            monitor="val_sensitivity",
            mode="max",
        )
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
                best_metrics_callback,
                checkpoint_callback,
                #early_stopping,
            ],
            log_every_n_steps=1,
            enable_progress_bar=True,
        )

        trainer.fit(model=model, datamodule=datamodule)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.flush()

        if not checkpoint_callback.best_model_path:
            raise RuntimeError(f"Fold {fold} did not produce a best checkpoint.")

        if best_metrics_callback.best_metrics is None:
            raise RuntimeError(
                f"Fold {fold} did not record metrics for the best checkpoint."
            )
        best_metrics = best_metrics_callback.best_metrics

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

        fold_metrics = {
            k: float(v) for k, v in best_metrics.items()
            if isinstance(v, (int, float)) or hasattr(v, "item")
        }
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
        best_score = checkpoint_callback.best_model_score
        fold_metrics["best_val_sensitivity_checkpoint_score"] = (
            float(best_score) if best_score is not None else float("nan")
        )

        print(f"\nFold {fold} metrics:")
        for k, v in fold_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")

        metrics_per_fold.append(fold_metrics)

        # Trainer owns the optimizer, loop state, and accelerator references.
        # Release them before constructing the next fold.
        del (
            trainer,
            model,
            datamodule,
            tabular_model,
            tabular_preprocessor,
            tabular_feature_selector,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_dir = logs_root / model_name / "cross_validation_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if feature_selection_reports:
        selection_output_dir = checkpoints_root / model_name
        by_fold_path = selection_output_dir / "lasso_feature_selection_by_fold.csv"
        combined_report = pd.concat(feature_selection_reports, ignore_index=True)
        combined_report.to_csv(by_fold_path, index=False)

        stability_report = (
            combined_report.groupby(
                ["feature_name", "feature_group"],
                as_index=False,
            )
            .agg(
                feature_occurrence_count=("fold", "nunique"),
                selection_count=("selected", "sum"),
                mean_lasso_importance=("lasso_importance", "mean"),
                max_lasso_importance=("lasso_importance", "max"),
                mean_lasso_rank=("lasso_rank", "mean"),
            )
            .assign(
                selection_frequency=lambda report: (
                    report["selection_count"] / len(feature_selection_reports)
                )
            )
            .sort_values(
                [
                    "selection_frequency",
                    "mean_lasso_importance",
                    "feature_name",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
        )
        stability_path = selection_output_dir / "lasso_feature_stability.csv"
        stability_report.to_csv(stability_path, index=False)
        stability_chart_path = (
            selection_output_dir / "lasso_feature_stability.png"
        )
        save_lasso_feature_importance_chart(
            stability_report,
            stability_chart_path,
            title="Cross-fold LASSO feature importance and stability",
            importance_column="mean_lasso_importance",
            top_n=tabular_feature_plot_top_n,
            frequency_column="selection_frequency",
        )
        print(f"[LASSO] Fold selections: {by_fold_path}", flush=True)
        print(f"[LASSO] Cross-fold stability: {stability_path}", flush=True)
        print(f"[LASSO] Cross-fold chart: {stability_chart_path}", flush=True)

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

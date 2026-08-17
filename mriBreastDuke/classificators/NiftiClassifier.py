import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassAUROC,
    MulticlassROC,
    MulticlassRecall,
)
import matplotlib.pyplot as plt
import numpy as np

class DebugBatchShapeCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[TRAIN] batch {batch_idx} shape: {tuple(x.shape)}", flush=True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[VAL]   batch {batch_idx} shape: {tuple(x.shape)}", flush=True)



class NiftiClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes: int,
        lr=1e-3,
        class_names=None,
        class_weights=None,
        sensitivity_lambda: float = 0.3,
        sensitivity_class_weights=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        self.sensitivity_lambda = sensitivity_lambda
        self.eps = eps

        # Register weights as buffers so they move automatically to GPU/CPU with the model.
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        # Optional: weights for sensitivity penalty per class.
        # Useful when one class sensitivity is more important.
        if sensitivity_class_weights is not None:
            sensitivity_class_weights = torch.tensor(
                sensitivity_class_weights,
                dtype=torch.float32,
            )
        self.register_buffer("sensitivity_class_weights", sensitivity_class_weights)

        # Stateful validation metrics
        self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_roc = MulticlassROC(num_classes=num_classes)
        self.val_sensitivity = MulticlassRecall(
            num_classes=num_classes,
            average=None,
        )
        # Loss history for charts
        self.train_loss_history = []
        self.train_ce_loss_history = []
        self.train_sensitivity_loss_history = []
        self.train_soft_sensitivity_history = []

        self.val_loss_history = []
        self.val_ce_loss_history = []
        self.val_sensitivity_loss_history = []
        self.val_soft_sensitivity_history = []
        # Temporary epoch buffers
        self.train_epoch_losses = []
        self.train_epoch_ce_losses = []
        self.train_epoch_sensitivity_losses = []
        self.train_epoch_soft_sensitivities = []

        self.val_epoch_losses = []
        self.val_epoch_ce_losses = []
        self.val_epoch_sensitivity_losses = []
        self.val_epoch_soft_sensitivities = []

    def forward(self, x):
        return self.model(x)

    def sensitivity_aware_cross_entropy(self, logits, y):
        """
        Sensitivity-aware loss.

        If num_classes == 2:
            Uses binary cross-entropy with logits + soft binary sensitivity penalty.

            Supports:
                logits shape [B]
                logits shape [B, 1]
                logits shape [B, 2]

        If num_classes > 2:
            Uses multiclass cross-entropy + differentiable macro-sensitivity penalty.
        """

        # -------------------------
        # Binary classification
        # -------------------------
        if self.num_classes == 2:
            y_float = y.float()

            # Case A: model outputs one logit: [B] or [B, 1]
            if logits.ndim == 1 or logits.shape[1] == 1:
                binary_logits = logits.view(-1)

                # For BCEWithLogitsLoss, use pos_weight, not class_weights.
                # If class_weights = [weight_for_0, weight_for_1],
                # then pos_weight should usually be weight_for_1 / weight_for_0.
                pos_weight = None
                if self.class_weights is not None:
                    pos_weight = (
                            self.class_weights[1] / (self.class_weights[0] + self.eps)
                    ).view(1)

                ce_loss = F.binary_cross_entropy_with_logits(
                    binary_logits,
                    y_float,
                    pos_weight=pos_weight,
                )

                probs_positive = torch.sigmoid(binary_logits)

            # Case B: model outputs two logits: [B, 2]
            else:
                ce_loss = F.cross_entropy(
                    logits,
                    y.long(),
                    weight=self.class_weights,
                )

                probs_positive = torch.softmax(logits, dim=1)[:, 1]

            # Sensitivity = TP / (TP + FN)
            # Soft TP = sum(y * p_positive)
            # Support = number of actual positives
            positive_support = y_float.sum()

            if positive_support > 0:
                soft_tp = (y_float * probs_positive).sum()
                soft_sensitivity = soft_tp / (positive_support + self.eps)
            else:
                # No positive samples in this batch.
                # Do not penalize sensitivity for this batch.
                soft_sensitivity = torch.tensor(
                    1.0,
                    device=logits.device,
                    dtype=logits.dtype,
                )

            sensitivity_loss = 1.0 - soft_sensitivity

            total_loss = ce_loss + self.sensitivity_lambda * sensitivity_loss

            return total_loss, ce_loss, sensitivity_loss, soft_sensitivity

        # -------------------------
        # Multiclass classification
        # -------------------------
        ce_loss = F.cross_entropy(
            logits,
            y.long(),
            weight=self.class_weights,
        )

        probs = torch.softmax(logits, dim=1)

        y_one_hot = F.one_hot(y.long(), num_classes=self.num_classes).float()

        soft_tp_per_class = (y_one_hot * probs).sum(dim=0)
        support_per_class = y_one_hot.sum(dim=0)

        present_classes = support_per_class > 0

        soft_sensitivity_per_class = soft_tp_per_class / (
                support_per_class + self.eps
        )

        if self.sensitivity_class_weights is not None:
            class_weights = self.sensitivity_class_weights.to(logits.device)

            valid_weights = class_weights[present_classes]
            valid_sensitivities = soft_sensitivity_per_class[present_classes]

            macro_soft_sensitivity = (
                                             valid_sensitivities * valid_weights
                                     ).sum() / (valid_weights.sum() + self.eps)
        else:
            macro_soft_sensitivity = soft_sensitivity_per_class[
                present_classes
            ].mean()

        sensitivity_loss = 1.0 - macro_soft_sensitivity

        total_loss = ce_loss + self.sensitivity_lambda * sensitivity_loss

        return total_loss, ce_loss, sensitivity_loss, macro_soft_sensitivity

    def _step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)

        loss, ce_loss, sensitivity_loss, soft_sensitivity = (
            self.sensitivity_aware_cross_entropy(logits, y)
        )

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_ce_loss", ce_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            f"{stage}_sensitivity_loss",
            sensitivity_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_soft_sensitivity",
            soft_sensitivity,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Save batch values for custom charts
        if stage == "train":
            self.train_epoch_losses.append(loss.detach())
            self.train_epoch_ce_losses.append(ce_loss.detach())
            self.train_epoch_sensitivity_losses.append(sensitivity_loss.detach())
            self.train_epoch_soft_sensitivities.append(soft_sensitivity.detach())
        elif stage == "val":
            self.val_epoch_losses.append(loss.detach())
            self.val_epoch_ce_losses.append(ce_loss.detach())
            self.val_epoch_sensitivity_losses.append(sensitivity_loss.detach())
            self.val_epoch_soft_sensitivities.append(soft_sensitivity.detach())

        return loss, logits, y
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch, stage="val")

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.val_cm.update(preds, y)
        self.val_auc.update(probs, y)
        self.val_roc.update(probs, y)
        self.val_sensitivity.update(preds, y)

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_epoch_losses) > 0:
            val_loss = torch.stack(self.val_epoch_losses).mean().detach().cpu().item()
            val_ce_loss = torch.stack(self.val_epoch_ce_losses).mean().detach().cpu().item()
            val_sensitivity_loss = (
                torch.stack(self.val_epoch_sensitivity_losses).mean().detach().cpu().item()
            )
            val_soft_sensitivity = (
                torch.stack(self.val_epoch_soft_sensitivities).mean().detach().cpu().item()
            )

            self.val_loss_history.append(val_loss)
            self.val_ce_loss_history.append(val_ce_loss)
            self.val_sensitivity_loss_history.append(val_sensitivity_loss)
            self.val_soft_sensitivity_history.append(val_soft_sensitivity)

            fig_losses = self._fig_loss_curves()
            self._tb_add_figure("losses/loss_curves", fig_losses)
            plt.close(fig_losses)

            fig_soft_sensitivity = self._fig_soft_sensitivity_curve()
            self._tb_add_figure("losses/soft_sensitivity_curve", fig_soft_sensitivity)
            plt.close(fig_soft_sensitivity)

            self.val_epoch_losses.clear()
            self.val_epoch_ce_losses.clear()
            self.val_epoch_sensitivity_losses.clear()
            self.val_epoch_soft_sensitivities.clear()
        # --- Scalars ---
        auc = self.val_auc.compute()
        # Oncotype classification treats class 1 as positive.  Checkpointing
        # therefore uses its recall rather than macro recall over both classes.
        per_class_recall = self.val_sensitivity.compute()
        sensitivity = per_class_recall[1]

        self.log("val_auc_roc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_sensitivity", sensitivity, prog_bar=True, on_step=False, on_epoch=True)

        # --- Confusion matrix figure ---
        cm = self.val_cm.compute().detach().cpu().numpy()
        fig_cm = self._fig_confusion_matrix(cm, self.class_names)
        self._tb_add_figure("val/confusion_matrix", fig_cm)
        plt.close(fig_cm)

        # --- ROC curve figure (one-vs-rest per class) ---
        fpr, tpr, thresholds = self.val_roc.compute()
        self._print_roc_points(fpr, tpr, thresholds, self.class_names)
        fig_roc = self._fig_multiclass_roc(fpr, tpr, self.class_names)
        self._tb_add_figure("val/roc_curve", fig_roc)
        plt.close(fig_roc)

        # reset
        self.val_cm.reset()
        self.val_auc.reset()
        self.val_roc.reset()
        self.val_sensitivity.reset()

    def on_train_epoch_end(self):
        if len(self.train_epoch_losses) == 0:
            return

        train_loss = torch.stack(self.train_epoch_losses).mean().detach().cpu().item()
        train_ce_loss = torch.stack(self.train_epoch_ce_losses).mean().detach().cpu().item()
        train_sensitivity_loss = (
            torch.stack(self.train_epoch_sensitivity_losses).mean().detach().cpu().item()
        )
        train_soft_sensitivity = (
            torch.stack(self.train_epoch_soft_sensitivities).mean().detach().cpu().item()
        )

        self.train_loss_history.append(train_loss)
        self.train_ce_loss_history.append(train_ce_loss)
        self.train_sensitivity_loss_history.append(train_sensitivity_loss)
        self.train_soft_sensitivity_history.append(train_soft_sensitivity)

        self.train_epoch_losses.clear()
        self.train_epoch_ce_losses.clear()
        self.train_epoch_sensitivity_losses.clear()
        self.train_epoch_soft_sensitivities.clear()

    def _fig_loss_curves(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        epochs_train = range(1, len(self.train_loss_history) + 1)
        epochs_val = range(1, len(self.val_loss_history) + 1)

        if len(self.train_loss_history) > 0:
            ax.plot(epochs_train, self.train_loss_history, label="train total loss")
            ax.plot(epochs_train, self.train_ce_loss_history, label="train CE loss")
            ax.plot(
                epochs_train,
                self.train_sensitivity_loss_history,
                label="train sensitivity loss",
            )

        if len(self.val_loss_history) > 0:
            ax.plot(epochs_val, self.val_loss_history, label="val total loss")
            ax.plot(epochs_val, self.val_ce_loss_history, label="val CE loss")
            ax.plot(
                epochs_val,
                self.val_sensitivity_loss_history,
                label="val sensitivity loss",
            )

        ax.set_title("Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        return fig

    def _fig_soft_sensitivity_curve(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        epochs_train = range(1, len(self.train_soft_sensitivity_history) + 1)
        epochs_val = range(1, len(self.val_soft_sensitivity_history) + 1)

        if len(self.train_soft_sensitivity_history) > 0:
            ax.plot(
                epochs_train,
                self.train_soft_sensitivity_history,
                label="train soft sensitivity",
            )

        if len(self.val_soft_sensitivity_history) > 0:
            ax.plot(
                epochs_val,
                self.val_soft_sensitivity_history,
                label="val soft sensitivity",
            )

        ax.set_title("Soft Sensitivity Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Soft Sensitivity")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

        fig.tight_layout()
        return fig

    @staticmethod
    def _print_roc_points(fpr, tpr, thresholds, class_names):
        print("[VAL] ROC curve points:", flush=True)
        for c, name in enumerate(class_names):
            fpr_values = fpr[c].detach().cpu().numpy()
            tpr_values = tpr[c].detach().cpu().numpy()
            thr_values = thresholds[c].detach().cpu().numpy()
            print(
                f"  class={name} | thresholds={thr_values.tolist()} | "
                f"fpr={fpr_values.tolist()} | tpr={tpr_values.tolist()}",
                flush=True,
            )

    def _tb_add_figure(self, tag: str, fig):
        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(tag, fig, global_step=self.current_epoch)

    @staticmethod
    def _fig_confusion_matrix(cm: np.ndarray, class_names):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        thresh = cm.max() * 0.6 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )

        fig.tight_layout()
        return fig

    @staticmethod
    def _fig_multiclass_roc(fpr, tpr, class_names):
        fig, ax = plt.subplots(figsize=(7, 6))
        for c, name in enumerate(class_names):
            ax.plot(
                fpr[c].detach().cpu().numpy(),
                tpr[c].detach().cpu().numpy(),
                label=name,
            )

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

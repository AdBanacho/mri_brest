import torch
from torch import nn
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAUROC, MulticlassROC
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
    def __init__(self, model, num_classes: int, lr=1e-3, class_names=None):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        # Stateful validation metrics
        self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_roc = MulticlassROC(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

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

        return loss

    def on_validation_epoch_end(self):
        # --- Scalars ---
        auc = self.val_auc.compute()
        self.log("val_auc_roc", auc, prog_bar=True, on_step=False, on_epoch=True)

        # --- Confusion matrix figure ---
        cm = self.val_cm.compute().detach().cpu().numpy()
        fig_cm = self._fig_confusion_matrix(cm, self.class_names)
        self._tb_add_figure("val/confusion_matrix", fig_cm)
        plt.close(fig_cm)

        # --- ROC curve figure (one-vs-rest per class) ---
        fpr, tpr, _ = self.val_roc.compute()  # each: [C, ...]
        fig_roc = self._fig_multiclass_roc(fpr, tpr, self.class_names)
        self._tb_add_figure("val/roc_curve", fig_roc)
        plt.close(fig_roc)

        # reset
        self.val_cm.reset()
        self.val_auc.reset()
        self.val_roc.reset()

    def _tb_add_figure(self, tag: str, fig):
        # Works with TensorBoardLogger: self.logger.experiment is a SummaryWriter
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

        # write counts
        thresh = cm.max() * 0.6 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9
                )

        fig.tight_layout()
        return fig

    @staticmethod
    def _fig_multiclass_roc(fpr, tpr, class_names):
        fig, ax = plt.subplots(figsize=(7, 6))
        for c, name in enumerate(class_names):
            ax.plot(fpr[c].detach().cpu().numpy(), tpr[c].detach().cpu().numpy(), label=name)

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curves (One-vs-Rest)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
import torch
from torch import nn
import pytorch_lightning as pl

from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAUROC

class DebugBatchShapeCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[TRAIN] batch {batch_idx} shape: {tuple(x.shape)}", flush=True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[VAL]   batch {batch_idx} shape: {tuple(x.shape)}", flush=True)



class NiftiClassifier(pl.LightningModule):
    def __init__(self, model, num_classes: int, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

        # Validation metrics (stateful across batches)
        self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
        # For AUROC, we need probabilities (softmax), not argmax.

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

        # Update epoch-level metrics
        self.val_cm.update(preds, y)
        self.val_auc.update(probs, y)

        return loss

    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()          # shape: [C, C]
        auc = self.val_auc.compute()        # scalar (macro by default)

        # Log AUROC as a normal scalar metric
        self.log("val_auc_roc", auc, prog_bar=True, on_step=False, on_epoch=True)

        # Confusion matrix isn't a scalar; log it in a practical way:
        # 1) store as an attribute for inspection
        self.val_confusion_matrix = cm.detach().cpu()

        # 2) optionally log per-class accuracy derived from CM (scalar-friendly)
        per_class_acc = cm.diag() / cm.sum(dim=1).clamp_min(1)
        for i, v in enumerate(per_class_acc):
            self.log(f"val_acc_class_{i}", v, prog_bar=False, on_step=False, on_epoch=True)

        # Reset metric states for next epoch
        self.val_cm.reset()
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
import torch
from torch import nn

import pytorch_lightning as pl

class DebugBatchShapeCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[TRAIN] batch {batch_idx} shape: {tuple(x.shape)}", flush=True)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        print(f"[VAL]   batch {batch_idx} shape: {tuple(x.shape)}", flush=True)


class NiftiClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

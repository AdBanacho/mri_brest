from pytorch_lightning.loggers import TensorBoardLogger
from monai.networks.nets import DenseNet121
import torch
import pytorch_lightning as pl

from mriBreastDuke.dataLoaders import get_oncotype_score_for_series_as_serie_and_label_df, NiftiDataModule
from mriBreastDuke.constants import NIFTI_PATH, SEED, LIGHTING_LOGS
from mriBreastDuke.classificators import NiftiClassifier, DebugBatchShapeCallback, Simple3DFCN


def train(df, model_name, model):
    data_module = NiftiDataModule(
        df,
        # target_size=(256, 256, 64),
        image_root=NIFTI_PATH,
        batch_size=4,
        num_workers=4,
    )

    logger = TensorBoardLogger(
        save_dir=LIGHTING_LOGS,
        name=model_name
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[DebugBatchShapeCallback()],
        log_every_n_steps=1,
        enable_progress_bar=False
    )

    trainer.fit(model=model, datamodule=data_module)

    print("\n=== Final Validation Metrics ===")
    print(f"\n ===      {model_name}      ===")
    metrics = trainer.callback_metrics
    for k, v in metrics.items():
        print(f"{k}: {float(v):.4f}")


def main():
    df = get_oncotype_score_for_series_as_serie_and_label_df(50, 12, SEED)
    # df = get_oncotype_score_for_series_as_serie_and_label_df()

    num_classes = len(set(df.label))
    models = [('FCN',
               NiftiClassifier(Simple3DFCN(num_classes=num_classes))),
              ('DenseNet',
               NiftiClassifier(DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes)))]

    for model in models:
        train(df, *model)


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()
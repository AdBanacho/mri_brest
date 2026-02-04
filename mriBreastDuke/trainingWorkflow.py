from pytorch_lightning.loggers import TensorBoardLogger
from monai.networks.nets import DenseNet121
import torch
import pytorch_lightning as pl
import argparse
from monai.networks.nets import ViT

from mriBreastDuke.classificators.VistionTransformer import ViTOnlyLogits
from mriBreastDuke.dataLoaders import (get_oncotype_score_for_series_as_serie_and_label_df, NiftiDataModule,
                                       get_oncotype_score_for_series_as_studyId_and_label_df)
from mriBreastDuke.constants import NIFTI_PATH, SEED, LIGHTING_LOGS
from mriBreastDuke.classificators import NiftiClassifier, DebugBatchShapeCallback, Simple3DFCN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=int, default=0,
                        help="0=FCN, 1=DenseNet, 2=ViT3D")
    parser.add_argument("--epoch", type=int, default=50)

    return parser.parse_args()
    
def train(df, grouped_by_study, model_name, model):
    args = parse_args()
    data_module = NiftiDataModule(
        df,
        grouped_by_study,
        target_size=(256, 256, 64),
        image_root=NIFTI_PATH,
        batch_size=12,
        num_workers=4,
    )

    logger = TensorBoardLogger(
        save_dir=LIGHTING_LOGS,
        name=model_name
    )

    trainer = pl.Trainer(
        max_epochs=args.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[DebugBatchShapeCallback()],
        log_every_n_steps=1,
        enable_progress_bar=True
        )

    trainer.fit(model=model, datamodule=data_module)

    print("\n=== Final Validation Metrics ===")
    print(f"\n ===      {model_name}      ===")
    metrics = trainer.callback_metrics
    for k, v in metrics.items():
        print(f"{k}: {float(v):.4f}")


def main():
    # df = get_oncotype_score_for_series_as_serie_and_label_df(50, 12, SEED)
    df, grouped_by_study = get_oncotype_score_for_series_as_studyId_and_label_df()
    args = parse_args()
    num_classes = len(set(df.label))


    models = [
        ("FCN", NiftiClassifier(Simple3DFCN(num_classes=num_classes))),
        ("DenseNet", NiftiClassifier(DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_classes))),
        ("ViT3D_Base", ViTOnlyLogits(num_classes=num_classes)),
    ]

    train(df, grouped_by_study, *models[args.model])




if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

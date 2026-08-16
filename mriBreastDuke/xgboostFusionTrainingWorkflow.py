"""Explicit command-line entry point for MRI + XGBoost decision fusion."""

import pytorch_lightning as pl

from mriBreastDuke.constants import SEED
from mriBreastDuke.fusionTrainingWorkflow import main


if __name__ == "__main__":
    pl.seed_everything(SEED)
    main()

"""Late-fusion neural network for volumetric MRI and tabular predictors."""

import torch
from torch import nn


class FusionClassifier(nn.Module):
    """Fuse an MRI embedding with a clinical/radiomics embedding.

    The supplied ``image_encoder`` must return a tensor shaped
    ``(batch, image_feature_dim)``. The tabular branch expects the dense vectors
    produced by ``ClinicalFeaturePreprocessor``.
    """

    def __init__(
        self,
        image_encoder,
        image_feature_dim,
        clinical_input_dim,
        num_classes,
        clinical_hidden_dim=64,
        fusion_hidden_dim=128,
        dropout=0.30,
    ):
        super().__init__()
        if image_feature_dim < 1 or clinical_input_dim < 1:
            raise ValueError("Image and clinical feature dimensions must be positive.")

        self.image_encoder = image_encoder
        self.image_feature_dim = int(image_feature_dim)
        self.clinical_input_dim = int(clinical_input_dim)

        self.image_projection = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim),
            nn.Linear(self.image_feature_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(self.clinical_input_dim, clinical_hidden_dim),
            nn.LayerNorm(clinical_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(clinical_hidden_dim, clinical_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim + clinical_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, images, clinical_features):
        image_features = self.image_encoder(images)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, start_dim=1)
        if image_features.ndim != 2 or image_features.shape[1] != self.image_feature_dim:
            raise ValueError(
                "image_encoder returned an unexpected shape: "
                f"{tuple(image_features.shape)}; expected (batch, {self.image_feature_dim})."
            )

        clinical_features = clinical_features.float()
        if clinical_features.ndim != 2 or clinical_features.shape[1] != self.clinical_input_dim:
            raise ValueError(
                "clinical_features has an unexpected shape: "
                f"{tuple(clinical_features.shape)}; expected "
                f"(batch, {self.clinical_input_dim})."
            )

        image_embedding = self.image_projection(image_features)
        clinical_embedding = self.clinical_encoder(clinical_features)
        fused = torch.cat((image_embedding, clinical_embedding), dim=1)
        return self.fusion_head(fused)

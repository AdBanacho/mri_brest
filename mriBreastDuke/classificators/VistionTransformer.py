import torch.nn as nn
from monai.networks.nets import ViT

class ViTOnlyLogits(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=(256, 256, 64),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=True,
            num_classes=num_classes,
            dropout_rate=0.1,
        )

    def forward(self, x):
        out = self.vit(x)
        # MONAI ViT may return logits or (logits, extra)
        if isinstance(out, tuple):
            out = out[0]
        return out

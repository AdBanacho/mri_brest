import torch.nn.functional as F
from torch import nn

class Simple3DFCN(nn.Module):
    """
    Fully convolutional 3D network:
    - Conv3d + MaxPool3d blocks
    - 1x1x1 Conv3d to get num_classes channels
    - AdaptiveAvgPool3d(1) to aggregate over D,H,W
    No Linear layers.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        # 1x1x1 conv to map 32 feature channels -> num_classes
        self.classifier_conv = nn.Conv3d(32, num_classes, kernel_size=1)

        # Global adaptive average pooling to get (B, C, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Feature extractor
        x = self.pool(F.relu(self.conv1(x)))  # (B, 8, D/2, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 16, D/4, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 32, D/8, H/8, W/8)

        # Class logits per spatial location
        x = self.classifier_conv(x)           # (B, num_classes, D/8, H/8, W/8)

        # Global average pooling over D,H,W -> (B, num_classes, 1, 1, 1)
        x = self.global_pool(x)

        # Flatten to (B, num_classes)
        x = x.view(x.size(0), -1)
        return x
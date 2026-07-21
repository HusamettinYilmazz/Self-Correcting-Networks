
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import SeparableConv2d


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels: int, num_classes: int):
        super().__init__()

        # low-level feature projection (ResNet layer1)
        self.project_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # fusion head: use separable convs (includes BN + ReLU inside SeparableConv2d)
        self.refine = nn.Sequential(
            SeparableConv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            SeparableConv2d(256, 256, kernel_size=3, padding=1, bias=False),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor, target_size):
        low = self.project_low(low)

        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([high, low], dim=1)
        x = self.refine(x)
        x = self.classifier(x)

        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

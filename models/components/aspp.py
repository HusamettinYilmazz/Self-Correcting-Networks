import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv.
    This layer includes BatchNorm and ReLU to match prior sequential usage.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()

        # 1x1 branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # atrous branches (use separable convs for Xception variant)
        self.atrous6 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.atrous12 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.atrous18 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)

        # image pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        b1 = self.conv1x1(x)
        b2 = self.atrous6(x)
        b3 = self.atrous12(x)
        b4 = self.atrous18(x)

        b5 = self.global_pool(x)
        b5 = F.interpolate(b5, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.project(x)

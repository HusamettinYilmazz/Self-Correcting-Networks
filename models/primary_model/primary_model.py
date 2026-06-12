import torch
import torch.nn as nn

from ..components import SegmentationEncoder
from ..components import SegmentationDecoder

class PrimarySegmentationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "xception65", pretrained: bool = True):
        super().__init__()
        self.encoder = SegmentationEncoder(backbone, pretrained)
        self.decoder = SegmentationDecoder(self.encoder.low_level_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = self.encoder(x)
        return self.decoder(low, high, target_size=x.shape[-2:])

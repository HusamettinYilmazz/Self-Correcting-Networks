import torch
import torch.nn as nn

from .boundingbox_model import BoundingBoxEncoder
from ..components import SegmentationEncoder, SegmentationDecoder

class AncillarySegmentationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "xception65", pretrained: bool = True):
        super().__init__()
        self.encoder = SegmentationEncoder(backbone, pretrained)
        self.bb_encoder = BoundingBoxEncoder(
            num_classes,
            low_level_channels=self.encoder.low_level_channels,
            high_level_channels=256,
        )
        self.decoder = SegmentationDecoder(self.encoder.low_level_channels, num_classes)

    def forward(self, x: torch.Tensor, bb_mask: torch.Tensor) -> torch.Tensor:
        low, high = self.encoder(x)

        ## Fuse bounding box attention
        attn_low, attn_high = self.bb_encoder(bb_mask, low.shape[-2:], high.shape[-2:])
        low = low * attn_low
        high = high * attn_high

        return self.decoder(low, high, target_size=x.shape[-2:])

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

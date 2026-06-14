
import torch
import torch.nn as nn
import torchvision.models as tvm
import timm

from .aspp import ASPPModule

from .backbone_enums import BackBoneEnums

class DeepLabV3PlusEncoder(nn.Module):
    def __init__(self, backbone: str = "xception65", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == BackBoneEnums.RESNET101.value:
            base = tvm.resnet101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
            self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            aspp_in = 2048
            self.low_level_channels = 256
        
        elif backbone == BackBoneEnums.XCEPTION65.value:
            self.backbone = timm.create_model(
                "hf_hub:timm/xception65.tf_in1k",
                pretrained=pretrained,
                features_only=True,
                output_stride=16
            )

            self.low_idx = 1
            self.high_idx = -1

            self.low_level_channels = self.backbone.feature_info[self.low_idx]["num_chs"]
            aspp_in = self.backbone.feature_info[self.high_idx]["num_chs"]

        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented. Add it here.")

        self.aspp = ASPPModule(aspp_in, out_channels=256)

    def forward(self, x: torch.Tensor):
        if self.backbone_name == "xception65":
            feats = self.backbone(x)

            low = feats[self.low_idx]
            high = feats[self.high_idx]
            high = self.aspp(high)
        elif self.backbone_name == "resnet101":
            x = self.layer0(x)
            low = self.layer1(x)     ## (B, 256, H/4, W/4) passed to decoder
            x = self.layer2(low)
            x = self.layer3(x)
            x = self.layer4(x)
            high = self.aspp(x)      ## (B, 256, H/16, W/16)
        else:
            raise NotImplementedError
        
        return low, high

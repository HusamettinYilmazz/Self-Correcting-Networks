import torch
from ..self_correction_interface import SelfCorrectionModule

class NoCorrectionModule(SelfCorrectionModule):
    
    def forward(self, primary_logits: torch.Tensor, ancillary_logits: torch.Tensor) -> torch.Tensor:
        return ancillary_logits

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

import torch
from ..self_correction_interface import SelfCorrectionModule


class LinearCorrectionModule(SelfCorrectionModule):
    def __init__(self, alpha_start: float = 30.0, alpha_end: float = 0.5):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self._alpha = alpha_start

    @property
    def alpha(self) -> float:
        return self._alpha

    def set_alpha(self, alpha: float):
        self._alpha = alpha

    def compute_alpha(self, step: int, total_steps: int) -> float:
        ratio = step / max(total_steps - 1, 1)
        alpha = self.alpha_start * (self.alpha_end / self.alpha_start) ** ratio
        self._alpha = alpha
        return alpha

    def forward(self, primary_logits: torch.Tensor, ancillary_logits: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha
        blended_logits = (primary_logits + alpha * ancillary_logits) / (alpha + 1.0)
        return blended_logits

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

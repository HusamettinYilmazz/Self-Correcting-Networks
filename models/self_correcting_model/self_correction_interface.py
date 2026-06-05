from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class SelfCorrectionModule(ABC, nn.Module):
    
    @abstractmethod
    def forward(self, primary_logits: torch.Tensor, 
                ancillary_logits: torch.Tensor,) -> torch.Tensor:
        
        pass

    def freeze(self):
        
        pass

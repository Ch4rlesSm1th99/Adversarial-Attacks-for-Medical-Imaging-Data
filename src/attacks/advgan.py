# AdvGAN attack wrapper (inference)

from __future__ import annotations
from typing import Optional, Callable
import torch
import torch.nn as nn

from .base import Attack, clamp01

class AdvGAN(Attack):
    """
    Uses a trained generator G: x -> delta in [-1,1] via tanh.
    Args
      G: torch.nn.Module returning delta logits; expected shape B,C,H,W
      scale: multiplier, often eps in [0,1]
      mask_fn: optional callable (x,y)-> B,1,H,W
    Notes
      No training here. Generator must be loaded by caller.
    """
    def __init__(self, G: nn.Module, scale: float = 8/255,
                 mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        self.G = G
        self.scale = float(scale)
        self.mask_fn = mask_fn
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = torch.tanh(self.G(x)) * self.scale
            if self.mask_fn is not None:
                m = self.mask_fn(x, y)  # B,1,H,W
                d = d * m
            x_adv = clamp01(x + d)
        return x_adv.detach()

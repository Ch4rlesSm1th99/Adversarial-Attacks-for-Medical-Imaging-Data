# FGSM attack

from __future__ import annotations
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack, clamp01

class FGSM(Attack):
    """
    FGSM untargeted or targeted.
    Args
      eps: linf step
      targeted: None (untargeted) or int 0/1
      mask_fn: optional callable (x,y)-> B,1,H,W
    """
    def __init__(self, eps: float = 8/255, targeted: Optional[int] = None,
                 mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        self.eps = float(eps)
        self.targeted = targeted
        self.mask_fn = mask_fn

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x0 = x.detach()
        x_adv = x0.clone().requires_grad_(True)

        logits = model(x_adv)
        if self.targeted is None:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            sign = 1.0
        else:
            y_t = torch.full_like(y, float(self.targeted))
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            sign = -1.0

        g = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        step = sign * self.eps * torch.sign(g)

        if self.mask_fn is not None:
            m = self.mask_fn(x0, y)  # B,1,H,W
            step = step * m

        x_adv = clamp01(x0 + step).detach()
        return x_adv

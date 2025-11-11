# BIM and PGD attacks (linf)

from __future__ import annotations
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Attack, clamp01

def _pgd_linf(
    model: nn.Module,
    x: torch.Tensor, y: torch.Tensor,
    eps: float, alpha: float, steps: int,
    targeted: Optional[int],
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    random_start: bool
) -> torch.Tensor:
    x0 = x.detach()
    if random_start:
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x0.clone()
    x_adv = x_adv.detach()

    for _ in range(int(steps)):
        x_adv = x_adv.clone().requires_grad_(True)
        logits = model(x_adv)
        if targeted is None:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            sign = 1.0
        else:
            y_t = torch.full_like(y, float(targeted))
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            sign = -1.0

        g = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        step = sign * alpha * torch.sign(g)

        if mask_fn is not None:
            m = mask_fn(x0, y)  # B,1,H,W
            step = step * m

        x_adv = x_adv.detach() + step
        # project to linf ball around x0
        delta = (x_adv - x0).clamp(-eps, eps)
        x_adv = clamp01(x0 + delta).detach()

    return x_adv

class BIM(Attack):
    """
    Basic Iterative Method (no random start)
    Args
      eps: linf radius
      alpha: step size
      steps: iterations
      targeted: None or 0/1
      mask_fn: optional callable (x,y)-> B,1,H,W
    """
    def __init__(self, eps: float = 8/255, alpha: float = 2/255, steps: int = 10,
                 targeted: Optional[int] = None,
                 mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.targeted = targeted
        self.mask_fn = mask_fn

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pgd_linf(model, x, y, self.eps, self.alpha, self.steps, self.targeted, self.mask_fn, random_start=False)

class PGD(Attack):
    """
    PGD with random start
    Args
      eps: linf radius
      alpha: step size
      steps: iterations
      targeted: None or 0/1
      mask_fn: optional callable (x,y)-> B,1,H,W
      random_start: bool
    """
    def __init__(self, eps: float = 8/255, alpha: float = 2/255, steps: int = 10,
                 targeted: Optional[int] = None, random_start: bool = True,
                 mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.targeted = targeted
        self.random_start = bool(random_start)
        self.mask_fn = mask_fn

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _pgd_linf(model, x, y, self.eps, self.alpha, self.steps, self.targeted, self.mask_fn, random_start=self.random_start)

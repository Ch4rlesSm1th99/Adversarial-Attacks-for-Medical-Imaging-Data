# Attack base, registry, wrappers

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, Tuple, List, Type
import importlib
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------

class Attack:
    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

# ------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------

def bce_logits_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)

def bce_logits_loss_targeted(logits: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y_t)

def pred01(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) >= 0.5).float()

def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

def apply_spatial_mask(d: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None: return d
    return d * mask

def apply_channel_mask(d: torch.Tensor, ch_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if ch_mask is None: return d
    return d * ch_mask

def linf_project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    delta = (x_adv - x).clamp(-eps, eps)
    return x + delta

def l2_project(x_adv: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    d = x_adv - x
    flat = d.view(d.size(0), -1)
    n = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(n), (eps / n))
    d = (flat * scale).view_as(d)
    return x + d

def tv_l2(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())

# ------------------------------------------------------------
# Wrappers
# ------------------------------------------------------------

@dataclass
class MaskSpec:
    mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ch_mask: Optional[torch.Tensor] = None

@dataclass
class BudgetSpec:
    mode: str = "linf"
    value: float = 8/255
    per_area_scale: Optional[str] = None

@dataclass
class EOTSpec:
    n: int = 1
    jitter: int = 0
    rot_deg: float = 0.0
    p_hflip: float = 0.0

def _eot_aug(x: torch.Tensor, e: EOTSpec) -> torch.Tensor:
    if e.jitter > 0:
        pad = e.jitter
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        offs_x = torch.randint(-pad, pad+1, (x.size(0),), device=x.device)
        offs_y = torch.randint(-pad, pad+1, (x.size(0),), device=x.device)
        xs = []
        for i in range(x.size(0)):
            ox, oy = int(offs_x[i].item()), int(offs_y[i].item())
            xs.append(x[i:i+1, :, pad+oy: pad+oy+x.size(2)-2*pad, pad+ox: pad+ox+x.size(3)-2*pad])
        x = torch.cat(xs, dim=0)
    if e.p_hflip > 0:
        flip = torch.rand((x.size(0),), device=x.device) < e.p_hflip
        x = torch.where(flip.view(-1,1,1,1), torch.flip(x, dims=(3,)), x)
    if e.rot_deg and abs(e.rot_deg) > 1e-6:
        pass
    return x

class MaskedAttack(Attack):
    def __init__(self, base: Attack, m: MaskSpec, b: Optional[BudgetSpec] = None, eps_for_clip: Optional[float] = None):
        self.base = base
        self.m = m
        self.b = b
        self.eps_for_clip = eps_for_clip

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_base = self.base(model, x, y)
        d = x_base - x
        if self.m.mask_fn is not None:
            mask = self.m.mask_fn(x, y)
            d = apply_spatial_mask(d, mask)
        if self.m.ch_mask is not None:
            d = apply_channel_mask(d, self.m.ch_mask)
        x_adv = x + d
        if self.b is not None:
            if self.b.mode == "linf":
                if self.b.per_area_scale in ("per_area","per_sqrt_area"):
                    a = 1.0
                    if self.m.mask_fn is not None:
                        mk = self.m.mask_fn(x, y)
                        a = mk.mean().item()
                        a = max(1e-6, a)
                    val = self.b.value
                    if self.b.per_area_scale == "per_area": val = val / a
                    if self.b.per_area_scale == "per_sqrt_area": val = val / math.sqrt(a)
                    x_adv = linf_project(x_adv, x, val)
                else:
                    x_adv = linf_project(x_adv, x, self.b.value)
            elif self.b.mode == "l2":
                x_adv = l2_project(x_adv, x, self.b.value)
        x_adv = clamp01(x_adv)
        return x_adv.detach()

class EOTAttack(Attack):
    def __init__(self, base: Attack, spec: EOTSpec):
        self.base = base
        self.spec = spec

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.spec.n <= 1:
            return self.base(model, x, y)
        outs = []
        for _ in range(self.spec.n):
            xa = _eot_aug(x, self.spec)
            outs.append(self.base(model, xa, y))
        return outs[-1].detach()

# ------------------------------------------------------------
# Registry
# ------------------------------------------------------------

@dataclass
class AttackEntry:
    module: str
    cls_name: str
    brief: str = ""
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

class AttackRegistry:
    def __init__(self):
        self._map: Dict[str, AttackEntry] = {}

    def register(self, name: str, module: str, cls_name: str, brief: str = "", default_kwargs: Optional[Dict[str,Any]] = None):
        name = name.lower().strip()
        self._map[name] = AttackEntry(module=module, cls_name=cls_name, brief=brief, default_kwargs=(default_kwargs or {}))

    def resolve(self, name: str) -> AttackEntry:
        key = name.lower().strip()
        if key not in self._map:
            raise KeyError(f"attack not found: {name}")
        return self._map[key]

    def make(self, name: str, **kwargs) -> Attack:
        e = self.resolve(name)
        mod = importlib.import_module(e.module)
        if not hasattr(mod, e.cls_name):
            raise ImportError(f"class {e.cls_name} not in module {e.module}")
        cls = getattr(mod, e.cls_name)
        if not inspect.isclass(cls):
            raise TypeError(f"{e.cls_name} is not a class")
        ctor_kwargs = dict(e.default_kwargs)
        ctor_kwargs.update(kwargs)
        obj = cls(**ctor_kwargs)
        if not isinstance(obj, Attack):
            raise TypeError(f"{e.cls_name} must inherit Attack")
        return obj

    def names(self) -> List[str]:
        return sorted(self._map.keys())

REG = AttackRegistry()

# ------------------------------------------------------------
# Pre-register names -> modules
# ------------------------------------------------------------
# White-box grads
REG.register("fgsm", "src.attacks.fgsm", "FGSM", "fast sign", dict(eps=8/255, targeted=None, mask_fn=None))
REG.register("bim",  "src.attacks.pgd",  "BIM",  "iter fgsm", dict(eps=8/255, alpha=2/255, steps=10, targeted=None, mask_fn=None))
REG.register("pgd",  "src.attacks.pgd",  "PGD",  "linf pgd",  dict(eps=8/255, alpha=2/255, steps=10, targeted=None, mask_fn=None))
REG.register("mifgsm", "src.attacks.mifgsm", "MIFGSM", "momentum",  dict(eps=8/255, alpha=2/255, steps=10, momentum=1.0, targeted=None, mask_fn=None))
REG.register("nifgsm", "src.attacks.mifgsm", "NIFGSM", "nesterov",  dict(eps=8/255, alpha=2/255, steps=10, momentum=1.0, targeted=None, mask_fn=None))
REG.register("cw",     "src.attacks.cw",     "CWL2",   "cw l2",     dict(c=1.0, steps=100, lr=5e-2, confidence=0.0, targeted=None, mask_fn=None))
REG.register("deepfool", "src.attacks.deepfool", "DeepFoolBinary", "min perturb", dict(steps=50, overshoot=0.02, mask_fn=None))

# EOT
REG.register("eot_fgsm", "src.attacks.eot", "EOTFGSM", "eot fgsm", dict(eot_n=4, eps=8/255, alpha=8/255, jitter=2, rot=0.0, p_hflip=0.0))
REG.register("eot_pgd",  "src.attacks.eot", "EOTPGD",  "eot pgd",  dict(eot_n=4, eps=8/255, alpha=2/255, steps=10, jitter=2, rot=0.0, p_hflip=0.0))

# Spatial / geometric
REG.register("stadv",   "src.attacks.stadv",   "StAdv",   "spatial flow", dict(flow_eps=2.0, steps=30, tv=1e-3, mask_fn=None))
REG.register("elastic", "src.attacks.elastic", "Elastic", "nonrigid",     dict(grid=16, alpha=20.0, sigma=4.0, steps=20, mask_fn=None))

# Patch
REG.register("patch_universal", "src.attacks.patch", "UniversalPatch", "universal patch", dict(size_rel=(0.2,0.2), steps=1000, lr=0.1))

# Frequency / modality
REG.register("fft_pgd",   "src.attacks.fft",   "FFTPGD",   "lowfreq pgd", dict(eps=8/255, alpha=2/255, steps=10, band=(0.0,0.25), mask_fn=None))
REG.register("kspace_pgd","src.attacks.kspace","KSpacePGD","kspace pgd",  dict(eps=8/255, alpha=2/255, steps=10, frac=0.1, mask_fn=None))

# Black-box
REG.register("square",  "src.attacks.square",  "SquareAttack", "score-based", dict(eps=8/255, iters=5000, p=0.05, mask_fn=None))
REG.register("nes",     "src.attacks.nes",     "NESAttack",    "gradient-free", dict(eps=8/255, steps=300, sigma=0.001, samples=40, alpha=2/255, mask_fn=None))
REG.register("bandits", "src.attacks.bandits", "BanditsTD",    "bandits-td", dict(eps=8/255, steps=300, samples=20, alpha=2/255, mask_fn=None))
REG.register("hsj",     "src.attacks.hsj",     "HSJ",          "boundary", dict(steps=50, max_queries=10000))

# Suites
REG.register("autoattack", "src.attacks.autoattack", "AutoAttackWrapper", "AA suite", dict(eps=8/255, version="standard", subset=None))

# 3D variants
REG.register("pgd3d", "src.attacks.pgd3d", "PGD3D", "3d pgd", dict(eps=8/255, alpha=2/255, steps=10))

# AdvGAN wrapper (inference)
REG.register("advgan", "src.attacks.advgan", "AdvGAN", "generator", dict(G=None, scale=8/255, mask_fn=None))

# ------------------------------------------------------------
# Factory and build with wrappers
# ------------------------------------------------------------

@dataclass
class BuildSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    mask_spec: Optional[MaskSpec] = None
    budget: Optional[BudgetSpec] = None
    eot: Optional[EOTSpec] = None
    clamp: bool = True

def build_attack(spec: BuildSpec) -> Attack:
    base = REG.make(spec.name, **spec.params)
    atk: Attack = base
    if spec.eot is not None and spec.eot.n > 1:
        atk = EOTAttack(atk, spec.eot)
    if spec.mask_spec is not None or spec.budget is not None or spec.clamp:
        atk = MaskedAttack(atk, spec.mask_spec or MaskSpec(), spec.budget, eps_for_clip=None)
    return atk

# ------------------------------------------------------------
# Simple adapter
# ------------------------------------------------------------

def list_attacks() -> List[str]:
    return REG.names()

def make_mask_spec(mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                   ch_mask: Optional[torch.Tensor] = None) -> MaskSpec:
    return MaskSpec(mask_fn=mask_fn, ch_mask=ch_mask)

def make_budget(mode: str = "linf", value: float = 8/255, per_area_scale: Optional[str] = None) -> BudgetSpec:
    return BudgetSpec(mode=mode, value=value, per_area_scale=per_area_scale)

def make_eot(n: int = 1, jitter: int = 0, rot_deg: float = 0.0, p_hflip: float = 0.0) -> EOTSpec:
    return EOTSpec(n=n, jitter=jitter, rot_deg=rot_deg, p_hflip=p_hflip)

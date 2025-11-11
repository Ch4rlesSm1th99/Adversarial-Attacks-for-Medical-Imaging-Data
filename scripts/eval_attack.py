#!/usr/bin/env python3
# Evaluate adversarial attacks for BraTS slice classifier

from __future__ import annotations
import argparse, json, math, os, random, time, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------------------------------
# utils
# ---------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_float_expr(s: str) -> float:
    s = s.strip()
    if "/" in s:
        a,b = s.split("/",1)
        return float(a) / float(b)
    return float(s)

def parse_eps_list(s: str) -> List[float]:
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t: continue
        out.append(parse_float_expr(t))
    if not out: raise ValueError("empty eps list")
    return out

def sigmoid01(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)

def tv_l2(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())

def ssim_simple(x: torch.Tensor, y: torch.Tensor, C1=0.01**2, C2=0.03**2, win_size=7) -> torch.Tensor:
    pad = win_size // 2
    ch = x.size(1)
    weight = torch.ones((ch,1,win_size,win_size), device=x.device, dtype=x.dtype) / (win_size*win_size)
    mu_x = F.conv2d(x, weight, padding=pad, groups=ch)
    mu_y = F.conv2d(y, weight, padding=pad, groups=ch)
    sigma_x = F.conv2d(x*x, weight, padding=pad, groups=ch) - mu_x*mu_x
    sigma_y = F.conv2d(y*y, weight, padding=pad, groups=ch) - mu_y*mu_y
    sigma_xy = F.conv2d(x*y, weight, padding=pad, groups=ch) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()

def save_grid_png(x: torch.Tensor, x_adv: torch.Tensor, diff: torch.Tensor, out_png: Path, n: int = 8):
    if plt is None:
        return
    n = min(n, x.size(0))
    def to_np(img): return img.detach().cpu().numpy()
    x0 = to_np(x[:n,0])
    xa0 = to_np(x_adv[:n,0])
    df0 = to_np(diff[:n,0])
    rows = n
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2*rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(rows):
        axes[i,0].imshow(x0[i], vmin=0.0, vmax=1.0, cmap="gray"); axes[i,0].axis("off")
        axes[i,1].imshow(xa0[i], vmin=0.0, vmax=1.0, cmap="gray"); axes[i,1].axis("off")
        df = (df0[i] - df0[i].min()) / (df0[i].ptp()+1e-6)
        axes[i,2].imshow(df, cmap="magma"); axes[i,2].axis("off")
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=120); plt.close(fig)

# ---------------------------------------
# data
# ---------------------------------------

class Cache2DDataset(Dataset):
    def __init__(self, cache_root: Path, names: List[str], split: str,
                 limit_subjects: Optional[int]=None, max_slices: Optional[int]=None, seed: int=1337):
        self.cache_root = cache_root
        self.names = names[:]
        if limit_subjects: self.names = self.names[:limit_subjects]
        self.items: List[Tuple[str,int,int]] = []
        rng = np.random.default_rng(seed)
        for name in self.names:
            with np.load(cache_root / f"{name}.npz") as z:
                seg = z["seg"]; D = seg.shape[2]
            pos = [int(k) for k in range(D) if (seg[...,k] > 0).any()]
            neg = [int(k) for k in range(D) if not (seg[...,k] > 0).any()]
            pairs = [(name, z, 1) for z in pos] + [(name, z, 0) for z in neg]
            self.items += pairs
        if max_slices:
            rng.shuffle(self.items)
            self.items = self.items[:max_slices]

    def __len__(self): return len(self.items)

    def __getitem__(self, i: int) -> Dict[str,Any]:
        name, z, lab = self.items[i]
        with np.load(self.cache_root / f"{name}.npz") as d:
            X = d["X"].astype(np.float32) / 65535.0  # 4,H,W,D
            seg = d["seg"][..., z].astype(np.uint8)
        x = X[..., z]
        return {"x": torch.from_numpy(x), "y": torch.tensor([float(lab)], dtype=torch.float32),
                "subject": name, "z": torch.tensor(int(z), dtype=torch.int32),
                "seg": torch.from_numpy(seg)}

class Shard2DDataset(Dataset):
    def __init__(self, shard_dir: Path, split: str, cache_root: Optional[Path]=None,
                 limit_files: Optional[int]=None, limit_rows: Optional[int]=None):
        self.files = sorted(shard_dir.glob(f"shard_{split}_*.npz"))
        if limit_files:
            self.files = self.files[:limit_files]
        self.cache_root = cache_root
        self.index: List[Tuple[int,int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = z["X"].shape[0]
            m = n if limit_rows is None else min(n, limit_rows)
            self.index += [(fi, j) for j in range(m)]

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int) -> Dict[str,Any]:
        fi, j = self.index[i]
        f = self.files[fi]
        with np.load(f) as z:
            X = z["X"][j].astype(np.float32)
            y = float(z["y"][j])
            subject = str(z["subject"][j])
            zidx = int(z["z"][j])
        if self.cache_root is not None:
            with np.load(self.cache_root / f"{subject}.npz") as d:
                seg = d["seg"][..., zidx].astype(np.uint8)
        else:
            seg = np.zeros((X.shape[1], X.shape[2]), dtype=np.uint8)
        return {"x": torch.from_numpy(X), "y": torch.tensor([y], dtype=torch.float32),
                "subject": subject, "z": torch.tensor(zidx, dtype=torch.int32),
                "seg": torch.from_numpy(seg)}

def discover_mode(data_root: Path) -> str:
    if any(data_root.glob("shard_val_*.npz")) or any(data_root.glob("shard_train_*.npz")):
        return "shard"
    if any(data_root.glob("BraTS2021_*.npz")):
        return "cache"
    raise SystemExit(f"[ERR] unknown data_root layout: {data_root}")

def load_splits(splits_path: Path) -> Dict[str,List[str]]:
    return json.loads(Path(splits_path).read_text())

def make_loader(data_root: Path, splits_path: Optional[Path], split: str, batch: int, workers: int,
                cache_root: Optional[Path], smoke: bool, limit_subjects: Optional[int], max_slices: Optional[int]) -> DataLoader:
    mode = discover_mode(data_root)
    if mode == "shard":
        ds = Shard2DDataset(data_root, split, cache_root=cache_root,
                            limit_files=(3 if smoke else None),
                            limit_rows=(800 if smoke else None))
    else:
        assert splits_path is not None and splits_path.exists(), "need splits.json for cache mode"
        names = load_splits(splits_path)[split]
        ds = Cache2DDataset(data_root, names, split,
                            limit_subjects=(limit_subjects if smoke else None),
                            max_slices=(max_slices if smoke else None))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=(workers>0))
    return dl

# ---------------------------------------
# models
# ---------------------------------------

class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.cv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)
    def forward(self, x): return self.rl(self.bn(self.cv(x)))

class UNetG(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.down1 = nn.Sequential(ConvBNRelu(in_ch, base), ConvBNRelu(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(ConvBNRelu(base, base*2), ConvBNRelu(base*2, base*2))
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = nn.Sequential(ConvBNRelu(base*2, base*4), ConvBNRelu(base*4, base*4))
        self.pool3 = nn.MaxPool2d(2)
        self.mid   = nn.Sequential(ConvBNRelu(base*4, base*8), ConvBNRelu(base*8, base*8))
        self.up3   = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3  = nn.Sequential(ConvBNRelu(base*8, base*4), ConvBNRelu(base*4, base*4))
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2  = nn.Sequential(ConvBNRelu(base*4, base*2), ConvBNRelu(base*2, base*2))
        self.up1   = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1  = nn.Sequential(ConvBNRelu(base*2, base), ConvBNRelu(base, base))
        self.out   = nn.Conv2d(base, in_ch, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m  = self.mid(self.pool3(d3))
        u3 = self.up3(m)
        c3 = torch.cat([u3, d3], dim=1); c3 = self.dec3(c3)
        u2 = self.up2(c3)
        c2 = torch.cat([u2, d2], dim=1); c2 = self.dec2(c2)
        u1 = self.up1(c2)
        c1 = torch.cat([u1, d1], dim=1); c1 = self.dec1(c1)
        out = self.out(c1)
        return out

class TinyCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),    nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

def load_frozen_target(ckpt_path: Path, device: torch.device, in_ch: int=4) -> nn.Module:
    try:
        from src.models.target import load_frozen_target as _load
        m = _load(str(ckpt_path), device)
        return m
    except Exception:
        m = TinyCNN(in_ch=in_ch).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        if "model" in state: state = state["model"]
        m.load_state_dict(state, strict=False)
        m.eval()
        for p in m.parameters(): p.requires_grad = False
        return m

def load_generator(gen_ckpt: Path, in_ch: int, device: torch.device) -> nn.Module:
    try:
        from src.models.advgan_gen import AdvGANGenerator
        G = AdvGANGenerator(in_ch=in_ch).to(device)
    except Exception:
        G = UNetG(in_ch=in_ch).to(device)
    ck = torch.load(str(gen_ckpt), map_location=device)
    sd = ck.get("G", None)
    if sd is None:
        sd = ck.get("state_dict", None)
    if sd is None:
        # try root
        sd = {k.replace("module.", ""): v for k,v in ck.items() if isinstance(v, torch.Tensor)}
    G.load_state_dict(sd, strict=False)
    G.eval()
    for p in G.parameters(): p.requires_grad = False
    return G

# ---------------------------------------
# masks
# ---------------------------------------

def mask_from_seg(seg: torch.Tensor, kind: str, ring: int) -> torch.Tensor:
    # seg: N x H x W uint8 or float
    seg = (seg > 0).float()
    if kind == "none": return torch.ones_like(seg)
    if kind == "seg": return seg
    if kind == "ring":
        if ring <= 0: return seg
        k = torch.ones((1,1,3,3), device=seg.device)
        dil = seg.unsqueeze(1)
        ero = seg.unsqueeze(1)
        for _ in range(ring):
            dil = (F.conv2d(dil, k, padding=1) > 0).float()
            ero = (F.conv2d(ero, k, padding=1) >= 9).float()
        ring_m = (dil - ero).clamp(0,1).squeeze(1)
        ring_m = ring_m * (1.0 - seg)
        any_ring = (ring_m.sum(dim=(1,2)) > 0).float().view(-1,1,1)
        return ring_m + (1.0 - any_ring) * seg
    return torch.ones_like(seg)

# ---------------------------------------
# attacks
# ---------------------------------------

def attack_fgsm(x: torch.Tensor, y: torch.Tensor, model: nn.Module, eps: float, targeted: Optional[int]=None,
                mask: Optional[torch.Tensor]=None, clip_min=0.0, clip_max=1.0) -> torch.Tensor:
    x = x.detach(); x.requires_grad_(True)
    logits = model(x)
    if targeted is None:
        loss = F.binary_cross_entropy_with_logits(logits, y)
        grad_sign = torch.sign(torch.autograd.grad(loss, x)[0])
        x_adv = x + eps * grad_sign
    else:
        tgt = torch.full_like(y, float(targeted))
        loss = F.binary_cross_entropy_with_logits(logits, tgt)
        grad_sign = torch.sign(torch.autograd.grad(loss, x)[0])
        x_adv = x - eps * grad_sign
    delta = x_adv - x
    if mask is not None:
        delta = delta * mask.unsqueeze(1)
    x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    return x_adv

def attack_pgd(x: torch.Tensor, y: torch.Tensor, model: nn.Module, eps: float, alpha: float, steps: int,
               targeted: Optional[int]=None, rand_start: bool=True, mask: Optional[torch.Tensor]=None,
               clip_min=0.0, clip_max=1.0) -> torch.Tensor:
    x = x.detach()
    if rand_start:
        delta = torch.empty_like(x).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x)
    if mask is not None:
        delta = delta * mask.unsqueeze(1)
    x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        if targeted is None:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + alpha * torch.sign(grad)
        else:
            tgt = torch.full_like(y, float(targeted))
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv - alpha * torch.sign(grad)
        delta = (x_adv - x).clamp(-eps, eps)
        if mask is not None:
            delta = delta * mask.unsqueeze(1)
        x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    return x_adv

def attack_bim(x: torch.Tensor, y: torch.Tensor, model: nn.Module, eps: float, alpha: float, steps: int,
               targeted: Optional[int]=None, mask: Optional[torch.Tensor]=None,
               clip_min=0.0, clip_max=1.0) -> torch.Tensor:
    return attack_pgd(x, y, model, eps=eps, alpha=alpha, steps=steps, targeted=targeted, rand_start=False,
                      mask=mask, clip_min=clip_min, clip_max=clip_max)

# ---------------------------------------
# eval
# ---------------------------------------

@dataclass
class Args:
    data_root: str
    cache_root: Optional[str]
    splits: Optional[str]
    split: str
    baseline_ckpt: str
    gen_ckpt: Optional[str]
    method: str
    eps: str
    alpha: Optional[str]
    steps: int
    targeted: Optional[int]
    mask: str
    ring: int
    batch: int
    workers: int
    out_dir: str
    grid: bool
    grid_n: int
    csv: str
    per_sample_csv: Optional[str]
    seed: int
    smoke: bool
    limit_subjects: Optional[int]
    max_slices: Optional[int]
    amp: bool

class Evaluator:
    def __init__(self, a: Args):
        self.a = a
        set_seed(a.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = Path(a.data_root)
        self.cache_root = Path(a.cache_root) if a.cache_root else None
        self.splits_path = Path(a.splits) if a.splits else None
        self.out_dir = Path(a.out_dir); ensure_dir(self.out_dir)
        self.fig_dir = self.out_dir / "figures"; ensure_dir(self.fig_dir)

        # loader
        self.dl = make_loader(self.data_root, self.splits_path, a.split, a.batch, a.workers,
                              cache_root=self.cache_root, smoke=a.smoke,
                              limit_subjects=a.limit_subjects, max_slices=a.max_slices)

        # sample for shape
        sm = next(iter(self.dl))
        self.in_ch = sm["x"].shape[1]

        # models
        self.T = load_frozen_target(Path(a.baseline_ckpt), self.device, in_ch=self.in_ch)
        self.G = None
        if a.method == "advgan":
            if not a.gen_ckpt:
                raise SystemExit("[ERR] --gen_ckpt required for method=advgan")
            self.G = load_generator(Path(a.gen_ckpt), self.in_ch, self.device)

        # eps list
        self.eps_list = parse_eps_list(a.eps)

        # alpha
        if a.alpha is None:
            self.alpha_list = [max(e/4.0, 1.0/255.0) for e in self.eps_list]
        else:
            al = parse_eps_list(a.alpha)
            if len(al) == 1 and len(self.eps_list) > 1:
                self.alpha_list = [al[0] for _ in self.eps_list]
            elif len(al) == len(self.eps_list):
                self.alpha_list = al
            else:
                raise SystemExit("[ERR] alpha list must be len 1 or match eps list")

        # logs
        self.csv_path = Path(a.csv); ensure_dir(self.csv_path.parent)
        self.sample_csv_path = Path(a.per_sample_csv) if a.per_sample_csv else None
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["eps","alpha","steps","clean_acc","robust_acc","asr","l2_mean","linf_mean","tv_mean","ssim_mean","n","clean_correct"])
        self.csv_file.flush()
        if self.sample_csv_path:
            ensure_dir(self.sample_csv_path.parent)
            self.sample_file = open(self.sample_csv_path, "w", newline="")
            self.sample_writer = csv.writer(self.sample_file)
            self.sample_writer.writerow(["subject","z","eps","pred_clean","pred_adv","y","flip","linf","l2","tv","ssim"])
            self.sample_file.flush()
        else:
            self.sample_file = None
            self.sample_writer = None

    @torch.no_grad()
    def pred01(self, logits: torch.Tensor) -> torch.Tensor:
        return (sigmoid01(logits) >= 0.5).float()

    def eval_once(self, eps: float, alpha: float, steps: int) -> Dict[str, float]:
        a = self.a
        amp = a.amp
        tot = 0
        clean_right = 0
        robust_right = 0
        flips = 0
        l2_sum = 0.0
        linf_sum = 0.0
        tv_sum = 0.0
        ssim_sum = 0.0
        clean_correct_seen = 0
        grid_saved = False

        for bi, batch in enumerate(self.dl, 1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            seg = batch["seg"].to(self.device, non_blocking=True)
            m = mask_from_seg(seg, a.mask, a.ring) if a.mask in ("seg","ring") else None

            # clean
            with torch.cuda.amp.autocast(enabled=amp):
                logits_c = self.T(x)
            pred_c = self.pred01(logits_c)
            clean_right += (pred_c == y).float().sum().item()

            # attack
            if a.method == "clean":
                x_adv = x.clone()
            elif a.method == "fgsm":
                x_adv = attack_fgsm(x, y, self.T, eps=eps, targeted=a.targeted, mask=m, clip_min=0.0, clip_max=1.0)
            elif a.method == "bim":
                x_adv = attack_bim(x, y, self.T, eps=eps, alpha=alpha, steps=steps, targeted=a.targeted, mask=m)
            elif a.method == "pgd":
                x_adv = attack_pgd(x, y, self.T, eps=eps, alpha=alpha, steps=steps, targeted=a.targeted, mask=m)
            elif a.method == "advgan":
                with torch.cuda.amp.autocast(enabled=amp):
                    delta = torch.tanh(self.G(x)) * eps
                if m is not None:
                    delta = delta * m.unsqueeze(1)
                x_adv = (x + delta).clamp(0.0, 1.0)
            else:
                raise SystemExit("bad method")

            with torch.cuda.amp.autocast(enabled=amp):
                logits_a = self.T(x_adv)
            pred_a = self.pred01(logits_a)

            robust_right += (pred_a == y).float().sum().item()
            cc = (pred_c == y).squeeze(1)
            clean_correct_seen += cc.sum().item()
            flips += (pred_a[cc] != y[cc]).float().sum().item()

            # stats
            d = (x_adv - x)
            l2_sum += float(d.view(d.size(0), -1).pow(2).sum(dim=1).sqrt().mean().item())
            linf_sum += float(d.abs().amax(dim=(1,2,3)).mean().item())
            tv_sum += float(tv_l2(d).item())
            ssim_sum += float(ssim_simple(x, x_adv).item())

            if a.grid and not grid_saved:
                save_grid_png(x, x_adv, d, self.fig_dir / f"{a.method}_eps{eps:.6f}.png", n=min(a.grid_n, x.size(0)))
                grid_saved = True

            if self.sample_writer is not None:
                for k in range(x.size(0)):
                    self.sample_writer.writerow([
                        str(batch["subject"][k]) if isinstance(batch["subject"], list) else "",
                        int(batch["z"][k].item()) if torch.is_tensor(batch["z"]) else 0,
                        f"{eps:.6f}",
                        int(pred_c[k].item()),
                        int(pred_a[k].item()),
                        int(y[k].item()),
                        int((pred_a[k].item() != y[k].item()) and (pred_c[k].item() == y[k].item())),
                        float(d[k].abs().amax().item()),
                        float(d[k].view(1, -1).pow(2).sum(dim=1).sqrt().item()),
                        float(tv_l2(d[k:k+1]).item()),
                        float(ssim_simple(x[k:k+1], x_adv[k:k+1]).item()),
                    ])

            tot += x.size(0)
            if a.smoke and tot >= 1024:
                break

        clean_acc = clean_right / max(1, tot)
        robust_acc = robust_right / max(1, tot)
        asr = flips / max(1, clean_correct_seen)
        n = tot
        l2_m = l2_sum / max(1, math.ceil(tot / self.a.batch))
        linf_m = linf_sum / max(1, math.ceil(tot / self.a.batch))
        tv_m = tv_sum / max(1, math.ceil(tot / self.a.batch))
        ssim_m = ssim_sum / max(1, math.ceil(tot / self.a.batch))
        return dict(eps=eps, alpha=alpha, steps=steps, clean_acc=clean_acc, robust_acc=robust_acc,
                    asr=asr, n=n, clean_correct=clean_correct_seen, l2_mean=l2_m, linf_mean=linf_m,
                    tv_mean=tv_m, ssim_mean=ssim_m)

    def run(self):
        for eps, alpha in zip(self.eps_list, self.alpha_list):
            m = self.eval_once(eps, alpha, self.a.steps)
            self.csv_writer.writerow([f"{m['eps']:.6f}", f"{m['alpha']:.6f}", m["steps"],
                                      f"{m['clean_acc']:.6f}", f"{m['robust_acc']:.6f}", f"{m['asr']:.6f}",
                                      f"{m['l2_mean']:.6f}", f"{m['linf_mean']:.6f}",
                                      f"{m['tv_mean']:.6f}", f"{m['ssim_mean']:.6f}", m["n"], m["clean_correct"]])
            self.csv_file.flush()
            print(f"[EPS {m['eps']:.6f}] clean={m['clean_acc']:.3f} robust={m['robust_acc']:.3f} asr={m['asr']:.3f}")

        self.csv_file.close()
        if self.sample_file:
            self.sample_file.close()

# ---------------------------------------
# cli
# ---------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="shard dir or subject cache dir")
    ap.add_argument("--cache_root", default=None, help="subject cache dir if shards used and masks need seg")
    ap.add_argument("--splits", default=str(repo / "data" / "processed" / "splits.json"))
    ap.add_argument("--split", choices=["train","val","test"], default="val")

    ap.add_argument("--baseline_ckpt", default=str(repo / "results" / "baseline" / "tinycnn.pt"))
    ap.add_argument("--gen_ckpt", default=None, help="AdvGAN G checkpoint for method=advgan")

    ap.add_argument("--method", choices=["clean","fgsm","bim","pgd","advgan"], default="advgan")
    ap.add_argument("--eps", type=str, default="0.0078,0.0117,0.0157")
    ap.add_argument("--alpha", type=str, default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--targeted", type=int, choices=[0,1], default=None)

    ap.add_argument("--mask", choices=["none","seg","ring"], default="none")
    ap.add_argument("--ring", type=int, default=2)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--out_dir", default=str(repo / "results" / "eval"))
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid_n", type=int, default=8)

    ap.add_argument("--csv", default=str(repo / "results" / "eval" / "summary.csv"))
    ap.add_argument("--per_sample_csv", default=None)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--max_slices", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    return ap

def main():
    args_ns = build_argparser().parse_args()
    # masks need seg
    if args_ns.mask in ("seg","ring"):
        if discover_mode(Path(args_ns.data_root)) == "shard" and not args_ns.cache_root:
            raise SystemExit("[ERR] --mask seg/ring requires --cache_root with subject .npz to access seg")
    a = Args(**vars(args_ns))
    ev = Evaluator(a)
    ev.run()
    print("[DONE] eval complete")

if __name__ == "__main__":
    main()

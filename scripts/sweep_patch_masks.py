#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, math, random, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from scripts.eval_attack import (
    make_loader, ensure_dir, load_frozen_target, load_generator,
    attack_fgsm, attack_bim, attack_pgd, sigmoid01
)

# -------------------------
# utils
# -------------------------

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_list_float(expr: str) -> List[float]:
    xs = []
    if not expr: return xs
    for t in expr.split(","):
        t = t.strip()
        if not t: continue
        if "/" in t:
            a,b = t.split("/",1)
            xs.append(float(a)/float(b))
        else:
            xs.append(float(t))
    return xs

def parse_list_int(expr: str) -> List[int]:
    xs = []
    if not expr: return xs
    for t in expr.split(","):
        t = t.strip()
        if not t: continue
        xs.append(int(t))
    return xs

def gaussian_kernel2d(ks: int, sigma: float, device):
    ax = torch.arange(ks, device=device) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    w = torch.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    w = w / w.sum()
    return w

def blur_mask(m: torch.Tensor, ks: int, sigma: float, power: float = 1.0) -> torch.Tensor:
    if ks <= 1 or sigma <= 0:
        return m.clamp(0,1)
    w = gaussian_kernel2d(ks, sigma, m.device).view(1,1,ks,ks)
    out = F.conv2d(m, w, padding=ks//2)
    if power != 1.0:
        out = out.clamp(0,1).pow(power)
    return out.clamp(0,1)

def morphologic(mask: torch.Tensor, iters: int, op: str) -> torch.Tensor:
    if iters <= 0: return mask
    k = torch.ones((1,1,3,3), device=mask.device)
    out = mask.clone()
    for _ in range(iters):
        if op == "dilate":
            out = (F.conv2d(out, k, padding=1) > 0).float()
        elif op == "erode":
            out = (F.conv2d(out, k, padding=1) >= 9).float()
    return out

def bbox_from_mask(mask2d: torch.Tensor) -> Tuple[int,int,int,int]:
    ys, xs = torch.where(mask2d > 0)
    if ys.numel() == 0:
        return 0, 0, 0, 0
    y0 = int(ys.min().item()); y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item()); x1 = int(xs.max().item()) + 1
    return x0, y0, x1 - x0, y1 - y0

def centroid_from_mask(mask2d: torch.Tensor) -> Tuple[int,int]:
    ys, xs = torch.where(mask2d > 0)
    if ys.numel() == 0: return 0, 0
    cx = int(xs.float().mean().item())
    cy = int(ys.float().mean().item())
    return cx, cy

def clamp_rect(x0: int, y0: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x0 = max(0, min(x0, W-1))
    y0 = max(0, min(y0, H-1))
    w  = max(1, min(w, W - x0))
    h  = max(1, min(h, H - y0))
    return x0, y0, w, h

# -------------------------
# patch config
# -------------------------

@dataclass
class PatchSpec:
    shape: str
    rel_w: Optional[float]
    rel_h: Optional[float]
    abs_w: Optional[int]
    abs_h: Optional[int]
    aspect: Optional[float]
    anchor: str
    cx: float
    cy: float
    jitter_px: int
    min_overlap: float
    max_overlap: float
    seg_label: str
    ring_width_px: int
    soft_ks: int
    soft_sigma: float
    soft_power: float
    combine_with_seg: str
    seg_dilate: int
    seg_erode: int
    multi_n: int
    multi_mode: str
    schedule_area: Optional[str]
    schedule_step: int
    allow_empty_on_fail: bool

def label_mask_from_seg(seg: torch.Tensor, label: str) -> torch.Tensor:
    if label == "any": return (seg > 0).float()
    if label == "et":  return (seg == 4).float()
    if label == "ed":  return (seg == 2).float()
    if label == "ncr": return (seg == 1).float()
    return (seg > 0).float()

def compute_patch_size(spec: PatchSpec, H: int, W: int) -> Tuple[int,int]:
    if spec.rel_w is not None and spec.rel_h is not None:
        w = max(1, int(round(spec.rel_w * W)))
        h = max(1, int(round(spec.rel_h * H)))
    elif spec.abs_w is not None and spec.abs_h is not None:
        w = max(1, int(spec.abs_w))
        h = max(1, int(spec.abs_h))
    elif spec.rel_w is not None and spec.aspect is not None:
        w = max(1, int(round(spec.rel_w * W)))
        h = max(1, int(round(w / max(1e-6, spec.aspect))))
    elif spec.rel_h is not None and spec.aspect is not None:
        h = max(1, int(round(spec.rel_h * H)))
        w = max(1, int(round(h * spec.aspect)))
    else:
        w = max(1, int(round(0.1 * W)))
        h = max(1, int(round(0.1 * H)))
    w = min(w, W); h = min(h, H)
    return w, h

def rect_mask(B: int, H: int, W: int, x0: int, y0: int, w: int, h: int, device) -> torch.Tensor:
    M = torch.zeros(B, 1, H, W, device=device)
    x0, y0, w, h = clamp_rect(x0, y0, w, h, W, H)
    M[:, :, y0:y0+h, x0:x0+w] = 1.0
    return M

def ellipse_mask(B: int, H: int, W: int, x0: int, y0: int, w: int, h: int, device) -> torch.Tensor:
    M = torch.zeros(B, 1, H, W, device=device)
    x0, y0, w, h = clamp_rect(x0, y0, w, h, W, H)
    yy = torch.arange(H, device=device).view(H,1).expand(H,W)
    xx = torch.arange(W, device=device).view(1,W).expand(H,W)
    cx = x0 + w / 2.0; cy = y0 + h / 2.0
    rx = max(1.0, w / 2.0); ry = max(1.0, h / 2.0)
    E = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    M[:, :, E] = 1.0
    return M

def ring_from_mask(inner: torch.Tensor, width_px: int) -> torch.Tensor:
    if width_px <= 0: return inner
    k = torch.ones((1,1,3,3), device=inner.device)
    dil = inner.clone()
    for _ in range(width_px):
        dil = (F.conv2d(dil, k, padding=1) > 0).float()
    return (dil - inner).clamp(0,1)

def iou_mask(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = ((a>0) & (b>0)).float().sum().item()
    union = ((a>0) | (b>0)).float().sum().item()
    return float(inter / (union + 1e-6))

def area_fraction(mask: torch.Tensor) -> float:
    B, _, H, W = mask.shape
    return float(mask.sum().item() / (B*H*W + 1e-6))

def soft_combine(m_patch: torch.Tensor, m_seg: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none": return m_patch
    if mode == "union": return (m_patch + m_seg).clamp(0,1)
    if mode == "intersect": return (m_patch * m_seg).clamp(0,1)
    if mode == "patch_minus_seg": return (m_patch * (1.0 - m_seg)).clamp(0,1)
    if mode == "seg_minus_patch": return (m_seg * (1.0 - m_patch)).clamp(0,1)
    return m_patch

def place_rect_xy(mode: str, spec: PatchSpec, H: int, W: int, w: int, h: int, seg_any: torch.Tensor) -> Tuple[int,int]:
    if mode == "center":
        cx = int(round(spec.cx * W)); cy = int(round(spec.cy * H))
        x0 = cx - w // 2; y0 = cy - h // 2
        return clamp_rect(x0, y0, w, h, W, H)[:2]
    if mode.startswith("grid:"):
        m = re.match(r"grid:(\d+),(\d+),(\d+)", mode)
        if m:
            rows = max(1, int(m.group(1))); cols = max(1, int(m.group(2))); idx = int(m.group(3))
            r = idx // cols; c = idx % cols
            cell_w = W // cols; cell_h = H // rows
            x0 = c*cell_w + max(0,(cell_w - w)//2)
            y0 = r*cell_h + max(0,(cell_h - h)//2)
            return clamp_rect(x0, y0, w, h, W, H)[:2]
    if mode == "random":
        x0 = random.randint(0, max(0, W - w))
        y0 = random.randint(0, max(0, H - h))
        return x0, y0
    if mode == "tumor_center":
        cx, cy = centroid_from_mask(seg_any)
        x0 = int(cx - w // 2); y0 = int(cy - h // 2)
        return clamp_rect(x0, y0, w, h, W, H)[:2]
    if mode == "tumor_bbox":
        x, y, bw, bh = bbox_from_mask(seg_any)
        if bw == 0 or bh == 0:
            return 0, 0
        cx = x + bw // 2; cy = y + bh // 2
        x0 = int(cx - w // 2); y0 = int(cy - h // 2)
        return clamp_rect(x0, y0, w, h, W, H)[:2]
    if mode == "outside_tumor":
        trials = 50
        for _ in range(trials):
            x0 = random.randint(0, max(0, W - w))
            y0 = random.randint(0, max(0, H - h))
            patch = torch.zeros(1,1,H,W, device=seg_any.device)
            patch[:,:,y0:y0+h,x0:x0+w] = 1.0
            if iou_mask(patch.squeeze(), seg_any) < 0.01:
                return x0, y0
        return 0, 0
    cx = int(round(spec.cx * W)); cy = int(round(spec.cy * H))
    x0 = cx - w // 2; y0 = cy - h // 2
    return clamp_rect(x0, y0, w, h, W, H)[:2]

def build_patch_mask(spec: PatchSpec, seg: torch.Tensor) -> torch.Tensor:
    B, H, W = seg.shape
    device = seg.device
    seg_any = label_mask_from_seg(seg, spec.seg_label)
    if spec.seg_dilate > 0:
        seg_any = morphologic(seg_any.unsqueeze(1), spec.seg_dilate, "dilate").squeeze(1)
    if spec.seg_erode > 0:
        seg_any = morphologic(seg_any.unsqueeze(1), spec.seg_erode, "erode").squeeze(1)

    if spec.schedule_area and spec.schedule_area.startswith("linear:"):
        m = re.match(r"linear:([^:]+):([^:]+):([^:]+)", spec.schedule_area)
        if m:
            a = float(m.group(1)); b = float(m.group(2)); steps = max(1, int(m.group(3)))
            t = min(max(spec.schedule_step, 0), steps-1) / max(1, steps-1)
            rel_area = (1.0 - t) * a + t * b
            if spec.aspect is not None:
                spec.rel_w = math.sqrt(rel_area * spec.aspect)
                spec.rel_h = spec.rel_w / max(1e-6, spec.aspect)
            else:
                spec.rel_w = math.sqrt(rel_area); spec.rel_h = spec.rel_w

    w, h = compute_patch_size(spec, H, W)
    m_out = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        seg_b = seg_any[b]
        x0, y0 = place_rect_xy(spec.anchor, spec, H, W, w, h, seg_b)
        if spec.jitter_px > 0:
            dx = random.randint(-spec.jitter_px, spec.jitter_px)
            dy = random.randint(-spec.jitter_px, spec.jitter_px)
            x0 = max(0, min(W-1, x0 + dx))
            y0 = max(0, min(H-1, y0 + dy))
        if spec.shape == "rect":
            patch = rect_mask(1, H, W, x0, y0, w, h, device)
        elif spec.shape == "ellipse":
            patch = ellipse_mask(1, H, W, x0, y0, w, h, device)
        elif spec.shape == "ring":
            base = ellipse_mask(1, H, W, x0, y0, w, h, device)
            patch = ring_from_mask(base, spec.ring_width_px)
        else:
            patch = rect_mask(1, H, W, x0, y0, w, h, device)

        if spec.min_overlap > 0 or spec.max_overlap < 1.0:
            iou = iou_mask(patch.squeeze(), seg_b)
            ok = True
            if spec.min_overlap > 0 and iou < spec.min_overlap: ok = False
            if spec.max_overlap < 1.0 and iou > spec.max_overlap: ok = False
            if not ok and spec.allow_empty_on_fail:
                patch = torch.zeros_like(patch)

        patch = soft_combine(patch, seg_b.unsqueeze(0).unsqueeze(0), spec.combine_with_seg)
        m_out[b:b+1] = patch

    if spec.soft_ks > 1 and spec.soft_sigma > 0:
        m_out = blur_mask(m_out, spec.soft_ks, spec.soft_sigma, spec.soft_power)

    if spec.multi_n and spec.multi_n > 1:
        acc = m_out.clone()
        for k in range(1, spec.multi_n):
            spec.schedule_step += 1
            acc = torch.maximum(acc, build_patch_mask(spec, seg))
        m_out = acc

    return m_out.clamp(0,1)

# -------------------------
# per-batch stats and bins
# -------------------------

def ssim_box(x: torch.Tensor, y: torch.Tensor, win: int = 7, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    ch = x.size(1); pad = win//2
    w = torch.ones((ch,1,win,win), device=x.device, dtype=x.dtype) / (win*win)
    mu_x = F.conv2d(x, w, padding=pad, groups=ch)
    mu_y = F.conv2d(y, w, padding=pad, groups=ch)
    sx = F.conv2d(x*x, w, padding=pad, groups=ch) - mu_x*mu_x
    sy = F.conv2d(y*y, w, padding=pad, groups=ch) - mu_y*mu_y
    sxy= F.conv2d(x*y, w, padding=pad, groups=ch) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y + C1) * (2*sxy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1) * (sx + sy + C2))
    return ssim.mean()

@dataclass
class BinAcc:
    edges: List[float]
    n: np.ndarray            # total tumour samples
    cc: np.ndarray           # clean correct tumour samples
    flips: np.ndarray        # flips among clean correct
    def as_rows(self, key_fields: List[Any]) -> List[List[Any]]:
        rows = []
        for i in range(len(self.edges)-1):
            lo = self.edges[i]; hi = self.edges[i+1]
            cc = int(self.cc[i]); fl = int(self.flips[i])
            asr = float(fl / max(1, cc))
            rows.append(key_fields + [f"{lo:.3f}", f"{hi:.3f}", int(self.n[i]), cc, fl, f"{asr:.6f}"])
        return rows

def init_bins(edges: List[float]) -> BinAcc:
    K = len(edges) - 1
    return BinAcc(edges=edges, n=np.zeros(K, dtype=np.int64), cc=np.zeros(K, dtype=np.int64), flips=np.zeros(K, dtype=np.int64))

def assign_bins(vals: np.ndarray, edges: List[float]) -> np.ndarray:
    idx = np.digitize(vals, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges)-2)
    return idx

# -------------------------
# run one setting
# -------------------------

def run_once(
    dl: DataLoader,
    T: torch.nn.Module,
    method: str,
    G: Optional[torch.nn.Module],
    eps: float,
    alpha: float,
    steps: int,
    spec: PatchSpec,
    amp: bool,
    limit_samples: Optional[int],
    save_first_grid: Optional[Path],
    grid_n: int,
    cov_edges: List[float]
) -> Tuple[Dict[str, float], BinAcc]:
    device = next(T.parameters()).device
    tot = 0
    clean_right = 0
    robust_right = 0
    flips = 0
    clean_correct_seen = 0

    linf_sum = 0.0
    l2_sum = 0.0
    tv_sum = 0.0
    ssim_sum = 0.0
    mask_area_sum = 0.0

    cov_vals = []
    cov_bins = init_bins(cov_edges)

    grid_done = False

    for bi, b in enumerate(dl, 1):
        x = b["x"].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        seg = b["seg"].to(device, non_blocking=True)
        B,C,H,W = x.shape

        spec.schedule_step = bi - 1
        m = build_patch_mask(spec, seg)    # B,1,H,W
        mask_area_sum += area_fraction(m) * B

        with torch.cuda.amp.autocast(enabled=amp):
            lc = T(x)
        pc = (torch.sigmoid(lc) >= 0.5).float()
        clean_right += (pc == y).float().sum().item()

        if method == "clean":
            xa = x.clone()
        elif method == "fgsm":
            xa = attack_fgsm(x, y, T, eps=eps, targeted=None, mask=m)
        elif method == "bim":
            xa = attack_bim(x, y, T, eps=eps, alpha=alpha, steps=steps, targeted=None, mask=m)
        elif method == "pgd":
            xa = attack_pgd(x, y, T, eps=eps, alpha=alpha, steps=steps, targeted=None, mask=m)
        elif method == "advgan":
            with torch.no_grad():
                d = torch.tanh(G(x)) * eps
            d = d * m
            xa = (x + d).clamp(0.0, 1.0)
        else:
            raise SystemExit("bad method")

        with torch.cuda.amp.autocast(enabled=amp):
            la = T(xa)
        pa = (torch.sigmoid(la) >= 0.5).float()

        robust_right += (pa == y).float().sum().item()
        cc = (pc == y).squeeze(1)
        clean_correct_seen += cc.sum().item()
        flips += (pa[cc] != y[cc]).float().sum().item()

        d = xa - x
        linf_sum += float(d.abs().amax(dim=(1,2,3)).mean().item())
        l2_sum   += float(d.view(d.size(0), -1).pow(2).sum(dim=1).sqrt().mean().item())
        dx = d[:,:,:,1:] - d[:,:,:,:-1]; dy = d[:,:,1:,:] - d[:,:,:-1,:]
        tv_sum += float(dx.pow(2).mean().item() + dy.pow(2).mean().item())
        ssim_sum += float(ssim_box(x, xa).item())

        seg_any = (seg > 0).float().unsqueeze(1)
        tumour_area = seg_any.sum(dim=(2,3)).squeeze(1)  # B
        inter = (m * seg_any).sum(dim=(2,3)).squeeze(1)  # B
        # coverage per sample on tumour slices
        mask = (tumour_area > 0)
        if mask.any():
            cov = (inter[mask] / tumour_area[mask].clamp(min=1.0)).detach().cpu().numpy()
            cov_vals.append(cov)
            # ASR bins count among clean-correct on tumour
            pc_mask = cc.detach().cpu().numpy().astype(bool)
            pa_np = pa.detach().cpu().numpy().squeeze(1).astype(int)
            y_np = y.detach().cpu().numpy().squeeze(1).astype(int)
            # iterate tumour slices only
            idxs = np.where(mask.detach().cpu().numpy())[0]
            if idxs.size > 0:
                cov_idx = assign_bins(cov, cov_edges)
                for k_local, k in enumerate(idxs):
                    # consider only if clean-correct
                    if pc_mask[k]:
                        cov_bins.cc[cov_idx[k_local]] += 1
                        if int(pa_np[k]) != int(y_np[k]):
                            cov_bins.flips[cov_idx[k_local]] += 1
                # counts of tumour slices
                for k_local in range(len(cov_idx)):
                    cov_bins.n[cov_idx[k_local]] += 1

        if (save_first_grid is not None) and (not grid_done):
            save_triplet_grid(x, xa, d, m, save_first_grid, n=min(grid_n, x.size(0)))
            grid_done = True

        tot += x.size(0)
        if limit_samples and tot >= limit_samples:
            break

    clean_acc = clean_right / max(1, tot)
    robust_acc = robust_right / max(1, tot)
    asr = flips / max(1, clean_correct_seen)
    n_batches = max(1, math.ceil(tot / dl.batch_size))
    cov_all = np.concatenate(cov_vals, axis=0) if len(cov_vals) else np.array([], dtype=np.float32)
    cov_mean = float(cov_all.mean()) if cov_all.size else 0.0
    cov_std  = float(cov_all.std()) if cov_all.size else 0.0

    out = dict(
        clean_acc=clean_acc, robust_acc=robust_acc, asr=asr, n=tot,
        linf_mean=linf_sum / n_batches, l2_mean=l2_sum / n_batches,
        tv_mean=tv_sum / n_batches, ssim_mean=ssim_sum / n_batches,
        mask_area_frac=mask_area_sum / max(1, tot),
        cov_mean=cov_mean, cov_std=cov_std
    )
    return out, cov_bins

# -------------------------
# small viz
# -------------------------

def save_triplet_grid(x: torch.Tensor, x_adv: torch.Tensor, diff: torch.Tensor, m: torch.Tensor, out_png: Path, n: int = 8):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = min(n, x.size(0))
    x0   = x[:n,0].detach().cpu().numpy()
    xa0  = x_adv[:n,0].detach().cpu().numpy()
    df0  = diff[:n,0].detach().cpu().numpy()
    ms0  = m[:n,0].detach().cpu().numpy()
    rows = n
    fig, axes = plt.subplots(rows, 4, figsize=(8, 2.2*rows))
    if rows == 1: axes = np.expand_dims(axes, 0)
    for i in range(rows):
        axes[i,0].imshow(x0[i], vmin=0.0, vmax=1.0, cmap="gray"); axes[i,0].axis("off")
        axes[i,1].imshow(xa0[i], vmin=0.0, vmax=1.0, cmap="gray"); axes[i,1].axis("off")
        df = (df0[i] - df0[i].min()) / (df0[i].ptp()+1e-6)
        axes[i,2].imshow(df, cmap="magma"); axes[i,2].axis("off")
        axes[i,3].imshow(ms0[i], vmin=0.0, vmax=1.0, cmap="viridis"); axes[i,3].axis("off")
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=120); plt.close(fig)

# -------------------------
# sweep driver
# -------------------------

@dataclass
class SweepArgs:
    data_root: str
    cache_root: Optional[str]
    splits: str
    split: str
    baseline_ckpt: str
    method: str
    gen_ckpt: Optional[str]
    eps: float
    alpha: float
    steps: int
    shape: str
    rel_ws: List[float]
    rel_hs: List[float]
    abs_ws: Optional[List[int]]
    abs_hs: Optional[List[int]]
    aspect_list: Optional[List[float]]
    anchor: str
    cx: float
    cy: float
    jitter_px: int
    min_overlap: float
    max_overlap: float
    seg_label: str
    ring_width_px: int
    soft_ks: int
    soft_sigma: float
    soft_power: float
    combine_with_seg: str
    seg_dilate: int
    seg_erode: int
    multi_n: int
    multi_mode: str
    schedule_area: Optional[str]
    replicates: int
    batch: int
    workers: int
    out_csv: str
    out_cov_csv: str
    out_fig_dir: str
    report_md: Optional[str]
    seed: int
    smoke: bool
    max_slices: Optional[int]
    amp: bool
    cov_bins: List[float]

def sweep(a: SweepArgs):
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(a.data_root)
    cache_root = Path(a.cache_root) if a.cache_root else None
    splits_path = Path(a.splits)
    out_csv = Path(a.out_csv); ensure_dir(out_csv.parent)
    out_cov_csv = Path(a.out_cov_csv) if a.out_cov_csv else out_csv.with_name(out_csv.stem + "_covbins.csv")
    ensure_dir(out_cov_csv.parent)
    fig_dir = Path(a.out_fig_dir); ensure_dir(fig_dir)

    dl = make_loader(
        data_root, splits_path, a.split, a.batch, a.workers,
        cache_root=(cache_root if cache_root else (data_root if any(data_root.glob("BraTS2021_*.npz")) else None)),
        smoke=a.smoke, limit_subjects=None, max_slices=a.max_slices
    )
    sm = next(iter(dl))
    in_ch = sm["x"].shape[1]

    T = load_frozen_target(Path(a.baseline_ckpt), device, in_ch=in_ch)
    G = None
    if a.method == "advgan":
        if not a.gen_ckpt: raise SystemExit("gen_ckpt required for method=advgan")
        G = load_generator(Path(a.gen_ckpt), in_ch, device)

    jobs: List[Tuple[Optional[int],Optional[int],Optional[float],Optional[float],Optional[float]]] = []
    if a.abs_ws and a.abs_hs:
        for aw in a.abs_ws:
            for ah in a.abs_hs:
                jobs.append((aw, ah, None, None, None))
    for rw in a.rel_ws:
        for rh in a.rel_hs:
            if a.aspect_list:
                for asp in a.aspect_list:
                    jobs.append((None, None, rw, rh, asp))
            else:
                jobs.append((None, None, rw, rh, None))

    # headers
    with open(out_csv, "w", newline="") as f_sum, open(out_cov_csv, "w", newline="") as f_cov:
        ws = csv.writer(f_sum)
        wc = csv.writer(f_cov)
        ws.writerow([
            "shape","rel_w","rel_h","abs_w","abs_h","aspect",
            "anchor","cx","cy","jitter_px",
            "min_overlap","max_overlap","seg_label","combine_with_seg",
            "soft_ks","soft_sigma","soft_power",
            "multi_n","multi_mode","schedule_area",
            "eps","alpha","steps",
            "replicate",
            "clean_acc","robust_acc","asr",
            "mask_area_frac","cov_mean","cov_std",
            "linf_mean","l2_mean","tv_mean","ssim_mean","n"
        ])
        wc.writerow([
            "shape","rel_w","rel_h","abs_w","abs_h","aspect",
            "anchor","combine_with_seg","eps","alpha","steps","replicate",
            "bin_lo","bin_hi","n_tumour","clean_correct","flips","asr_bin"
        ])

        for idx, (aw,ah,rw,rh,asp) in enumerate(jobs, start=1):
            for rep in range(a.replicates):
                spec = PatchSpec(
                    shape=a.shape,
                    rel_w=rw, rel_h=rh, abs_w=aw, abs_h=ah,
                    aspect=asp,
                    anchor=a.anchor, cx=a.cx, cy=a.cy,
                    jitter_px=a.jitter_px,
                    min_overlap=a.min_overlap, max_overlap=a.max_overlap,
                    seg_label=a.seg_label,
                    ring_width_px=a.ring_width_px,
                    soft_ks=a.soft_ks, soft_sigma=a.soft_sigma, soft_power=a.soft_power,
                    combine_with_seg=a.combine_with_seg,
                    seg_dilate=a.seg_dilate, seg_erode=a.seg_erode,
                    multi_n=a.multi_n, multi_mode=a.multi_mode,
                    schedule_area=a.schedule_area, schedule_step=0,
                    allow_empty_on_fail=True
                )

                tag = f"{a.shape}_rw{rw if rw is not None else -1:.3f}_rh{rh if rh is not None else -1:.3f}_aw{aw if aw else 0}_ah{ah if ah else 0}_asp{asp if asp else 0}"
                grid_png = fig_dir / f"{tag}.png"

                out, bins = run_once(
                    dl, T, a.method, G, a.eps, a.alpha, a.steps,
                    spec=spec, amp=a.amp,
                    limit_samples=(1200 if a.smoke else a.max_slices),
                    save_first_grid=grid_png, grid_n=8,
                    cov_edges=a.cov_bins
                )

                ws.writerow([
                    a.shape,
                    f"{rw:.6f}" if rw is not None else "",
                    f"{rh:.6f}" if rh is not None else "",
                    aw if aw is not None else "",
                    ah if ah is not None else "",
                    f"{asp:.6f}" if asp is not None else "",
                    a.anchor, f"{a.cx:.3f}", f"{a.cy:.3f}", a.jitter_px,
                    f"{a.min_overlap:.3f}", f"{a.max_overlap:.3f}", a.seg_label, a.combine_with_seg,
                    a.soft_ks, f"{a.soft_sigma:.3f}", f"{a.soft_power:.3f}",
                    a.multi_n, a.multi_mode, a.schedule_area if a.schedule_area else "",
                    f"{a.eps:.6f}", f"{a.alpha:.6f}", a.steps,
                    rep,
                    f"{out['clean_acc']:.6f}", f"{out['robust_acc']:.6f}", f"{out['asr']:.6f}",
                    f"{out['mask_area_frac']:.6f}", f"{out['cov_mean']:.6f}", f"{out['cov_std']:.6f}",
                    f"{out['linf_mean']:.6f}", f"{out['l2_mean']:.6f}", f"{out['tv_mean']:.6f}", f"{out['ssim_mean']:.6f}",
                    out["n"]
                ])
                for row in bins.as_rows([
                    a.shape,
                    f"{rw:.6f}" if rw is not None else "",
                    f"{rh:.6f}" if rh is not None else "",
                    aw if aw is not None else "",
                    ah if ah is not None else "",
                    f"{asp:.6f}" if asp is not None else "",
                    a.anchor, a.combine_with_seg,
                    f"{a.eps:.6f}", f"{a.alpha:.6f}", a.steps, rep
                ]):
                    wc.writerow(row)
                f_sum.flush(); f_cov.flush()

    if a.report_md:
        generate_report_md(
            sum_csv=out_csv,
            cov_csv=out_cov_csv,
            out_md=Path(a.report_md),
            top_k=5
        )

    print("[DONE] sweep complete ->", out_csv)

# -------------------------
# report maker
# -------------------------

def read_csv(path: Path) -> List[Dict[str,str]]:
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def group_key(row: Dict[str,str]) -> Tuple:
    return (
        row["shape"], row["rel_w"], row["rel_h"], row["abs_w"], row["abs_h"], row["aspect"],
        row["anchor"], row["combine_with_seg"], row["eps"], row["alpha"], row["steps"]
    )

def generate_report_md(sum_csv: Path, cov_csv: Path, out_md: Path, top_k: int = 5):
    ensure_dir(out_md.parent)
    S = read_csv(sum_csv)
    C = read_csv(cov_csv)

    # aggregate per setting across replicates
    agg: Dict[Tuple, Dict[str, Any]] = {}
    for r in S:
        k = group_key(r)
        a = agg.get(k, dict(asr=[], robust=[], clean=[], area=[], cov_mean=[], cov_std=[]))
        a["asr"].append(float(r["asr"]))
        a["robust"].append(float(r["robust_acc"]))
        a["clean"].append(float(r["clean_acc"]))
        a["area"].append(float(r["mask_area_frac"]))
        a["cov_mean"].append(float(r["cov_mean"]))
        a["cov_std"].append(float(r["cov_std"]))
        agg[k] = a

    def m(lst): return float(np.mean(lst)) if lst else 0.0
    def s(lst): return float(np.std(lst)) if lst else 0.0

    items = []
    for k, v in agg.items():
        items.append(dict(
            key=k,
            asr_mean=m(v["asr"]), asr_std=s(v["asr"]),
            robust_mean=m(v["robust"]), clean_mean=m(v["clean"]),
            area_mean=m(v["area"]), cov_mean=m(v["cov_mean"]), cov_std=m(v["cov_std"])
        ))
    items.sort(key=lambda x: x["asr_mean"], reverse=True)
    top = items[:top_k]

    # write md
    with open(out_md, "w") as f:
        f.write("# Patch sweep report\n\n")
        f.write("Summary of best configs by ASR.\n\n")
        f.write("| shape | rel_w | rel_h | abs_w | abs_h | aspect | anchor | comb | eps | steps | ASR | robust | clean | area | cov_mean |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for it in top:
            shape, rel_w, rel_h, abs_w, abs_h, aspect, anchor, comb, eps, alpha, steps = it["key"]
            f.write(f"| {shape} | {rel_w or ''} | {rel_h or ''} | {abs_w or ''} | {abs_h or ''} | {aspect or ''} | {anchor} | {comb} | {eps} | {steps} | {it['asr_mean']:.3f} | {it['robust_mean']:.3f} | {it['clean_mean']:.3f} | {it['area_mean']:.3f} | {it['cov_mean']:.3f} |\n")
        f.write("\n")

        # coverage bins tables for each top setting
        f.write("## ASR by tumour coverage bins\n\n")
        for it in top:
            shape, rel_w, rel_h, abs_w, abs_h, aspect, anchor, comb, eps, alpha, steps = it["key"]
            # gather cov rows
            rows = [r for r in C if (
                r["shape"]==shape and r["rel_w"]==rel_w and r["rel_h"]==rel_h and
                r["abs_w"]==abs_w and r["abs_h"]==abs_h and r["aspect"]==aspect and
                r["anchor"]==anchor and r["combine_with_seg"]==comb and
                r["eps"]==eps and r["steps"]==steps
            )]
            # aggregate across replicates
            bins: Dict[Tuple[str,str], Dict[str,int]] = {}
            for r in rows:
                key = (r["bin_lo"], r["bin_hi"])
                d = bins.get(key, dict(n=0, cc=0, flips=0))
                d["n"] += int(r["n_tumour"])
                d["cc"] += int(r["clean_correct"])
                d["flips"] += int(r["flips"])
                bins[key] = d
            # sort by bin_lo
            def lo(x): return float(x[0])
            kv = sorted(bins.items(), key=lambda kv: lo(kv[0]))
            f.write(f"### {shape} rel_w={rel_w or ''} rel_h={rel_h or ''} abs_w={abs_w or ''} abs_h={abs_h or ''} anchor={anchor}\n\n")
            f.write("| bin_lo | bin_hi | n_tumour | clean_correct | flips | ASR |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for (blo,bhi), d in kv:
                asr_bin = (d['flips'] / max(1, d['cc']))
                f.write(f"| {blo} | {bhi} | {d['n']} | {d['cc']} | {d['flips']} | {asr_bin:.3f} |\n")
            f.write("\n")

    print("[OK] report ->", out_md)

# -------------------------
# CLI
# -------------------------

def build_argparser():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_root", default=None)
    ap.add_argument("--splits", default=str(repo / "data" / "processed" / "splits.json"))
    ap.add_argument("--split", choices=["val","test"], default="val")

    ap.add_argument("--baseline_ckpt", default=str(repo / "results" / "baseline" / "tinycnn.pt"))
    ap.add_argument("--method", choices=["fgsm","bim","pgd","advgan","clean"], default="fgsm")
    ap.add_argument("--gen_ckpt", default=None)

    ap.add_argument("--eps", type=float, default=4/255)
    ap.add_argument("--alpha", type=float, default=1/255)
    ap.add_argument("--steps", type=int, default=10)

    ap.add_argument("--shape", choices=["rect","ellipse","ring"], default="rect")
    ap.add_argument("--rel_ws", type=str, default="0.05,0.10,0.20,0.30")
    ap.add_argument("--rel_hs", type=str, default="0.05,0.10,0.20,0.30")
    ap.add_argument("--abs_ws", type=str, default=None)
    ap.add_argument("--abs_hs", type=str, default=None)
    ap.add_argument("--aspect_list", type=str, default=None)

    ap.add_argument("--anchor", type=str, default="center")
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--jitter_px", type=int, default=0)

    ap.add_argument("--min_overlap", type=float, default=0.0)
    ap.add_argument("--max_overlap", type=float, default=1.1)
    ap.add_argument("--seg_label", choices=["any","et","ed","ncr"], default="any")
    ap.add_argument("--ring_width_px", type=int, default=6)

    ap.add_argument("--soft_ks", type=int, default=1)
    ap.add_argument("--soft_sigma", type=float, default=0.0)
    ap.add_argument("--soft_power", type=float, default=1.0)

    ap.add_argument("--combine_with_seg", choices=["none","union","intersect","patch_minus_seg","seg_minus_patch"], default="none")
    ap.add_argument("--seg_dilate", type=int, default=0)
    ap.add_argument("--seg_erode", type=int, default=0)

    ap.add_argument("--multi_n", type=int, default=1)
    ap.add_argument("--multi_mode", choices=["max","sum"], default="max")

    ap.add_argument("--schedule_area", type=str, default=None)

    ap.add_argument("--replicates", type=int, default=1)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--out_csv", default=str(repo / "results" / "eval" / "patch_sweep.csv"))
    ap.add_argument("--out_cov_csv", default=None)
    ap.add_argument("--out_fig_dir", default=str(repo / "results" / "figures" / "patch_sweep"))

    ap.add_argument("--report_md", default=str(repo / "results" / "eval" / "patch_sweep_report.md"))

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_slices", type=int, default=None)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--cov_bins", type=str, default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0")
    return ap

def main():
    args = build_argparser().parse_args()
    rel_ws = parse_list_float(args.rel_ws)
    rel_hs = parse_list_float(args.rel_hs)
    abs_ws = parse_list_int(args.abs_ws) if args.abs_ws else None
    abs_hs = parse_list_int(args.abs_hs) if args.abs_hs else None
    aspect = parse_list_float(args.aspect_list) if args.aspect_list else None
    cov_bins = parse_list_float(args.cov_bins)
    if len(cov_bins) < 2:
        cov_bins = [0.0, 1.0]

    a = SweepArgs(
        data_root=args.data_root, cache_root=args.cache_root, splits=args.splits, split=args.split,
        baseline_ckpt=args.baseline_ckpt, method=args.method, gen_ckpt=args.gen_ckpt,
        eps=float(args.eps), alpha=float(args.alpha), steps=int(args.steps),
        shape=args.shape,
        rel_ws=rel_ws, rel_hs=rel_hs, abs_ws=abs_ws, abs_hs=abs_hs, aspect_list=aspect,
        anchor=args.anchor, cx=float(args.cx), cy=float(args.cy), jitter_px=int(args.jitter_px),
        min_overlap=float(args.min_overlap), max_overlap=float(args.max_overlap),
        seg_label=args.seg_label, ring_width_px=int(args.ring_width_px),
        soft_ks=int(args.soft_ks), soft_sigma=float(args.soft_sigma), soft_power=float(args.soft_power),
        combine_with_seg=args.combine_with_seg, seg_dilate=int(args.seg_dilate), seg_erode=int(args.seg_erode),
        multi_n=int(args.multi_n), multi_mode=args.multi_mode,
        schedule_area=args.schedule_area,
        replicates=int(args.replicates),
        batch=int(args.batch), workers=int(args.workers),
        out_csv=args.out_csv, out_cov_csv=(args.out_cov_csv if args.out_cov_csv else None),
        out_fig_dir=args.out_fig_dir, report_md=args.report_md,
        seed=int(args.seed), smoke=bool(args.smoke), max_slices=args.max_slices, amp=bool(args.amp),
        cov_bins=cov_bins
    )
    sweep(a)

if __name__ == "__main__":
    main()

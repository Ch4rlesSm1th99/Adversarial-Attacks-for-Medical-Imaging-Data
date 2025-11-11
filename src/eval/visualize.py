#!/usr/bin/env python3
# Visualization utils for MRI slices, masks, diffs, curves

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any
import math
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# basic helpers
# --------------------------

DEFAULT_CH_NAMES = ["FLAIR", "T1", "T1CE", "T2"]

def _ensure_dir(p: Path):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo + 1e-6)

def _edge_from_mask(m: np.ndarray, th: int = 1) -> np.ndarray:
    # simple 4-neigh edges
    m = (m > 0).astype(np.uint8)
    e = np.zeros_like(m, dtype=np.uint8)
    e[1:,:] |= (m[1:,:] != m[:-1,:])
    e[:-1,:] |= (m[1:,:] != m[:-1,:])
    e[:,1:] |= (m[:,1:] != m[:,:-1])
    e[:,:-1] |= (m[:,1:] != m[:,:-1])
    if th > 1:
        from scipy.ndimage import binary_dilation
        e = binary_dilation(e, iterations=max(1, th-1)).astype(np.uint8)
    return e

def _rgba_overlay(base: np.ndarray, mask: np.ndarray,
                  color=(1.0, 0.0, 0.0), alpha=0.35) -> np.ndarray:
    """
    Overlay a binary mask onto a base image.

    base: gray [H,W] in [0,1] or RGBA [H,W,4]
    mask: bool or 0/1 array [H,W]
    color: RGB tuple in [0,1]
    alpha: overlay opacity
    """
    m = (mask > 0).astype(np.float32)
    if base.ndim == 2:
        out = np.stack([base, base, base, np.ones_like(base)], axis=-1)
    elif base.ndim == 3 and base.shape[-1] == 4:
        out = base.copy()
    else:
        raise ValueError("base must be gray [H,W] or RGBA [H,W,4]")
    r, g, b = color
    out[..., 0] = (1 - alpha * m) * out[..., 0] + alpha * m * r
    out[..., 1] = (1 - alpha * m) * out[..., 1] + alpha * m * g
    out[..., 2] = (1 - alpha * m) * out[..., 2] + alpha * m * b
    return out

def _select_channel(x: np.ndarray, ch: int) -> np.ndarray:
    # x: [C,H,W] or [B,C,H,W]
    if x.ndim == 4:
        return x[:, ch]
    return x[ch]

# --------------------------
# grids and mosaics
# --------------------------

def save_triplet_grid(x, x_adv, diff, out_path: Path, ch: int = 0, titles=("clean","adv","diff")):
    # x, x_adv, diff: [B,C,H,W] in [0,1]
    X = _to_np(x); Xa = _to_np(x_adv); Df = _to_np(diff)
    B = X.shape[0]; ch = int(ch)
    fig, axs = plt.subplots(B, 3, figsize=(9, 3*B))
    axs = np.array(axs, dtype=object).reshape(B, 3)
    for i in range(B):
        a = _clip01(_select_channel(X[i], ch))
        b = _clip01(_select_channel(Xa[i], ch))
        d = _norm01(_select_channel(Df[i], ch))
        axs[i,0].imshow(a, cmap="gray");  axs[i,0].set_title(titles[0]); axs[i,0].axis("off")
        axs[i,1].imshow(b, cmap="gray");  axs[i,1].set_title(titles[1]); axs[i,1].axis("off")
        axs[i,2].imshow(d, cmap="turbo"); axs[i,2].set_title(titles[2]); axs[i,2].axis("off")
    fig.tight_layout()
    out_path = _ensure_dir(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def save_modalities_grid(x, out_path: Path, ch_names: Sequence[str] = DEFAULT_CH_NAMES,
                         per_row: int = 4, vmin=0.0, vmax=1.0):
    # x: [C,H,W] or [B,C,H,W], first item used if batch
    X = _to_np(x)
    if X.ndim == 4:
        X = X[0]
    C = X.shape[0]
    ncol = int(per_row)
    nrow = int(math.ceil(C / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
    axs = np.array(axs, dtype=object).reshape(nrow, ncol)
    k = 0
    for r in range(nrow):
        for c in range(ncol):
            ax = axs[r, c]
            if k < C:
                im = _clip01(X[k])
                ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
                title = ch_names[k] if k < len(ch_names) else f"ch{k}"
                ax.set_title(title)
            ax.axis("off")
            k += 1
    fig.tight_layout()
    out_path = _ensure_dir(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def save_overlay_grid(x, out_path: Path, seg=None, mask=None, ch: int = 0,
                      seg_colors: Dict[int, Tuple[float,float,float]] = None,
                      alpha=0.35, show_outline=True, outline_th=1,
                      title="overlay"):
    # x: [C,H,W] or [B,C,H,W]
    # seg: [H,W] or [B,H,W], labels: 0,1,2,4
    # mask: [H,W] or [B,1,H,W]
    X = _to_np(x)
    B = 1 if X.ndim == 3 else X.shape[0]
    S = _to_np(seg) if seg is not None else None
    M = _to_np(mask) if mask is not None else None

    fig, axs = plt.subplots(B, 1, figsize=(4, 4*B))
    axs = np.array(axs, dtype=object).reshape(B, 1)

    if seg_colors is None:
        seg_colors = {1:(0.0,0.0,1.0), 2:(0.0,1.0,0.0), 4:(1.0,0.0,0.0)}

    for i in range(B):
        base = _clip01(_select_channel(X[i] if X.ndim == 4 else X, ch))
        rgba = np.stack([base, base, base, np.ones_like(base)], axis=-1)

        if S is not None:
            s2d = S[i] if (S.ndim == 3 and B > 1) else (S if S.ndim == 2 else S[0])
            s2d = s2d.astype(np.int32)
            for lbl, col in seg_colors.items():
                m_lbl = (s2d == lbl).astype(np.float32)
                rgba = _rgba_overlay(rgba, m_lbl, color=col, alpha=alpha)

        m = None
        if M is not None:
            if M.ndim == 4:
                m = M[i, 0]
            elif M.ndim == 3 and B > 1:
                m = M[i]
            else:
                m = M if M.ndim == 2 else M[0]
            rgba = _rgba_overlay(rgba, (m > 0).astype(np.float32), color=(1.0, 1.0, 0.0), alpha=0.25)

        axs[i, 0].imshow(rgba)
        if m is not None and show_outline:
            e = _edge_from_mask(m, th=int(outline_th))
            axs[i, 0].imshow(e, cmap="autumn", alpha=0.9, interpolation="nearest")
        axs[i, 0].set_title(title if B == 1 else f"{title} #{i}")
        axs[i, 0].axis("off")

    fig.tight_layout()
    out_path = _ensure_dir(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def save_batch_clean_adv_diff(x, x_adv, out_path: Path, seg=None,
                              ch: int = 0, alpha=0.30, max_rows: Optional[int] = None,
                              show_titles=True):
    # rows: each sample; cols: clean, adv, diff, seg
    X = _to_np(x); Xa = _to_np(x_adv)
    B = X.shape[0]
    R = min(B, max_rows) if max_rows else B
    fig, axs = plt.subplots(R, 4, figsize=(12, 3*R))
    axs = np.array(axs, dtype=object).reshape(R, 4)
    for i in range(R):
        a = _clip01(_select_channel(X[i], ch))
        b = _clip01(_select_channel(Xa[i], ch))
        d = _norm01(_select_channel(Xa[i] - X[i], ch))
        axs[i,0].imshow(a, cmap="gray"); axs[i,0].axis("off")
        axs[i,1].imshow(b, cmap="gray"); axs[i,1].axis("off")
        axs[i,2].imshow(d, cmap="turbo"); axs[i,2].axis("off")
        if seg is not None:
            s = _to_np(seg)
            s2d = s[i] if (s.ndim == 3) else s
            axs[i,3].imshow(a, cmap="gray")
            axs[i,3].imshow((s2d > 0).astype(np.float32), cmap="Reds", alpha=alpha)
            axs[i,3].axis("off")
        else:
            axs[i,3].imshow(a, cmap="gray"); axs[i,3].axis("off")
        if show_titles and i == 0:
            axs[i,0].set_title("clean")
            axs[i,1].set_title("adv")
            axs[i,2].set_title("diff")
            axs[i,3].set_title("seg")
    fig.tight_layout()
    out_path = _ensure_dir(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

# --------------------------
# eps curves and histos
# --------------------------

def save_curve(x_vals: Sequence[float], y_vals: Sequence[float],
               out_path: Path, xlabel="x", ylabel="y", title="curve"):
    fig = plt.figure(figsize=(5,4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    out_path = _ensure_dir(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def save_hist(values: Sequence[float], out_path: Path, bins=50, title="hist", xlabel="value"):
    fig = plt.figure(figsize=(5,4))
    plt.hist(values, bins=bins, density=True, alpha=0.8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("density")
    plt.grid(True, ls="--", alpha=0.3)
    out_path = _ensure_dir(out_path)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

def save_asr_vs_eps(curve_rows: List[Dict[str, float]], out_path: Path, title="ASR vs eps"):
    # rows: {eps, asr}
    eps = [r["eps"] for r in curve_rows]
    asr = [r["asr"] for r in curve_rows]
    save_curve(eps, asr, out_path, xlabel="eps", ylabel="ASR", title=title)

# --------------------------
# k-space and fft
# --------------------------

def save_fft_magnitude_grid(x, out_path: Path, ch: int = 0, log_scale=True):
    # x: [C,H,W] or [B,C,H,W]
    X = _to_np(x)
    if X.ndim == 4:
        X = X[0]
    im = _clip01(X[ch])
    F = np.fft.fftshift(np.fft.fft2(im))
    mag = np.abs(F)
    if log_scale:
        mag = np.log1p(mag)
        mag = _norm01(mag)
    fig, axs = plt.subplots(1, 2, figsize=(7,3))
    axs[0].imshow(im, cmap="gray"); axs[0].set_title("image"); axs[0].axis("off")
    axs[1].imshow(mag, cmap="magma"); axs[1].set_title("fft mag"); axs[1].axis("off")
    fig.tight_layout()
    out_path = _ensure_dir(out_path)
    fig.savefig(out_path, dpi=120); plt.close(fig)

# --------------------------
# selection helpers
# --------------------------

def select_examples_by_label(dataset, n_pos=4, n_neg=4) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # dataset[i] -> (x,y) or dict with x,y,seg
    pos_idx = []; neg_idx = []
    for i in range(len(dataset)):
        b = dataset[i]
        if isinstance(b, dict): y = _to_np(b["y"]).ravel()[0]
        else: y = _to_np(b[1]).ravel()[0]
        if y >= 0.5: pos_idx.append(i)
        else: neg_idx.append(i)
        if len(pos_idx) >= n_pos and len(neg_idx) >= n_neg:
            break
    return np.array(pos_idx), np.array(neg_idx), np.array(pos_idx + neg_idx)

def extract_batch(dataset, indices: Sequence[int]) -> Dict[str, Any]:
    xs = []; ys = []; segs = []
    for i in indices:
        b = dataset[i]
        if isinstance(b, dict):
            xs.append(_to_np(b["x"])); ys.append(_to_np(b["y"]))
            segs.append(_to_np(b.get("seg")) if ("seg" in b) else None)
        else:
            xs.append(_to_np(b[0])); ys.append(_to_np(b[1])); segs.append(None)
    X = np.stack(xs, 0); Y = np.stack(ys, 0)
    return dict(x=X, y=Y, seg=segs)


def save_sample_overlays(dataset, indices: Sequence[int], out_path: Path, ch: int = 0, title="overlay"):
    b = extract_batch(dataset, indices)
    x = b["x"]; seg = None
    # if some seg present, use first non None; else None
    if any(s is not None for s in b["seg"]):
        seg = np.stack([s if s is not None else np.zeros_like(x[0,0]) for s in b["seg"]], 0)
    save_overlay_grid(x, out_path=out_path, seg=seg, ch=ch, title=title)

def save_sample_triplets(dataset, indices: Sequence[int], x_adv: np.ndarray, out_path: Path, ch: int = 0):
    b = extract_batch(dataset, indices)
    x = b["x"]
    d = x_adv - x
    save_triplet_grid(x, x_adv, d, out_path=out_path, ch=ch, titles=("clean","adv","diff"))

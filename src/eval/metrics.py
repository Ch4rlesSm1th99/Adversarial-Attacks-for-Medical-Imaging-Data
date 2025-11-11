#!/usr/bin/env python3
# Binary metrics, norms, coverage, curves, aggregation

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Callable, Any
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# utils
# -----------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _sigmoid_np(z):
    z = _to_numpy(z)
    return 1.0 / (1.0 + np.exp(-z))

def _unpack_batch(b):
    # returns x, y, seg, subject, z
    if isinstance(b, dict):
        return b["x"], b["y"], b.get("seg", None), b.get("subject", None), b.get("z", None)
    # tuple (x,y)
    return b[0], b[1], None, None, None

def _ensure_01_probs(p):
    p = _to_numpy(p).astype(np.float64)
    if p.min() < 0.0 or p.max() > 1.0:
        p = 1.0 / (1.0 + np.exp(-p))
    return np.clip(p, 0.0, 1.0)

def _threshold(p, thr=0.5):
    return (p >= float(thr)).astype(np.int32)

# -----------------------------
# confusion and basic metrics
# -----------------------------

def confusion_from_preds(preds, y) -> Tuple[int,int,int,int]:
    pr = _to_numpy(preds).astype(np.int32).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    tp = int(((pr==1) & (gt==1)).sum())
    tn = int(((pr==0) & (gt==0)).sum())
    fp = int(((pr==1) & (gt==0)).sum())
    fn = int(((pr==0) & (gt==1)).sum())
    return tn, fp, fn, tp

def binary_metrics_from_probs(probs, y, thr=0.5) -> Dict[str, float]:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    pr = _threshold(p, thr)
    tn, fp, fn, tp = confusion_from_preds(pr, gt)
    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1   = (2*prec*rec) / max(1e-12, prec + rec)
    return dict(acc=acc, prec=prec, rec=rec, spec=spec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)

def best_f1_threshold(probs, y, grid=201) -> Tuple[float, Dict[str,float]]:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    thrs = np.linspace(0.0, 1.0, grid)
    best = (0.0, dict(acc=0, prec=0, rec=0, spec=0, f1=0, tp=0, tn=0, fp=0, fn=0))
    for t in thrs:
        m = binary_metrics_from_probs(p, gt, t)
        if m["f1"] > best[1]["f1"]:
            best = (float(t), m)
    return best

# -----------------------------
# curves and AUC
# -----------------------------

def roc_curve(probs, y) -> Tuple[np.ndarray,np.ndarray]:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    order = np.argsort(-p)  # desc
    tp = fp = 0
    P = int((gt==1).sum()); N = int((gt==0).sum())
    tpr = [0.0]; fpr = [0.0]
    for idx in order:
        if gt[idx] == 1: tp += 1
        else: fp += 1
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr)

def auc_trap(x, y) -> float:
    # x ascending
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    return float(np.trapz(y, x))

def roc_auc(probs, y) -> float:
    fpr, tpr = roc_curve(probs, y)
    return auc_trap(fpr, tpr)

def precision_recall_curve(probs, y) -> Tuple[np.ndarray,np.ndarray]:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    order = np.argsort(-p)  # desc
    tp = fp = 0
    P = int((gt==1).sum())
    prec = []; rec = []
    for idx in order:
        if gt[idx] == 1: tp += 1
        else: fp += 1
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, P)
        prec.append(precision); rec.append(recall)
    # add endpoints
    prec = np.array([1.0] + prec + [0.0])
    rec  = np.array([0.0] + rec + [1.0])
    return rec, prec

def pr_auc(probs, y) -> float:
    rec, prec = precision_recall_curve(probs, y)
    # area under P(R)
    return auc_trap(rec, prec)

# -----------------------------
# calibration
# -----------------------------

def ece(probs, y, n_bins=15) -> float:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    bins = np.linspace(0.0, 1.0, n_bins+1)
    e = 0.0; n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not m.any(): continue
        conf = p[m].mean()
        acc  = ( (p[m] >= 0.5).astype(np.int32) == gt[m] ).mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)

def brier(probs, y) -> float:
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.float64).ravel()
    return float(np.mean((p - gt) ** 2))

# -----------------------------
# norms and image deltas
# -----------------------------

def linf_mean(x, x_adv) -> float:
    a = _to_numpy(np.abs(x_adv - x))
    return float(a.reshape(a.shape[0], -1).max(axis=1).mean())

def l2_mean(x, x_adv) -> float:
    d = _to_numpy(x_adv - x)
    v = np.sqrt(np.sum(d*d, axis=(1,2,3)))
    return float(v.mean())

def l1_mean(x, x_adv) -> float:
    d = _to_numpy(np.abs(x_adv - x))
    v = np.sum(d, axis=(1,2,3))
    return float(v.mean())

def tv_mean(x, x_adv) -> float:
    d = torch.as_tensor(x_adv - x)
    dx = d[:,:,:,1:] - d[:,:,:,:-1]
    dy = d[:,:,1:,:] - d[:,:,:-1,:]
    return float((dx.pow(2).mean() + dy.pow(2).mean()).item())

def ssim_box(x, y, win=7, C1=0.01**2, C2=0.03**2) -> float:
    a = torch.as_tensor(x).float()
    b = torch.as_tensor(y).float()
    ch = a.size(1); pad = win//2
    w = torch.ones((ch,1,win,win), dtype=a.dtype) / (win*win)
    mu_a = F.conv2d(a, w, padding=pad, groups=ch)
    mu_b = F.conv2d(b, w, padding=pad, groups=ch)
    sa = F.conv2d(a*a, w, padding=pad, groups=ch) - mu_a*mu_a
    sb = F.conv2d(b*b, w, padding=pad, groups=ch) - mu_b*mu_b
    sab= F.conv2d(a*b, w, padding=pad, groups=ch) - mu_a*mu_b
    ssim = ((2*mu_a*mu_b + C1) * (2*sab + C2)) / ((mu_a*mu_a + mu_b*mu_b + C1) * (sa + sb + C2))
    return float(ssim.mean().item())

def perturb_stats(x, x_adv) -> Dict[str,float]:
    return dict(
        linf=linf_mean(x, x_adv),
        l2=l2_mean(x, x_adv),
        l1=l1_mean(x, x_adv),
        tv=tv_mean(x, x_adv),
        ssim=ssim_box(x, x_adv)
    )

# -----------------------------
# coverage and masks
# -----------------------------

def coverage_on_tumour(mask, seg) -> Tuple[np.ndarray, np.ndarray]:
    # mask, seg: B,1,H,W or B,H,W
    m = torch.as_tensor(mask).float()
    s = torch.as_tensor(seg).float()
    if m.dim()==3: m = m.unsqueeze(1)
    if s.dim()==3: s = s.unsqueeze(1)
    s_bin = (s > 0).float()
    inter = (m * s_bin).sum(dim=(2,3)).squeeze(1)   # B
    area_t = s_bin.sum(dim=(2,3)).squeeze(1)        # B
    cov = inter / torch.clamp(area_t, min=1.0)
    return cov.detach().cpu().numpy(), area_t.detach().cpu().numpy()

def iou_mask(mask, seg) -> float:
    m = torch.as_tensor(mask).float()
    s = torch.as_tensor(seg).float()
    if m.dim()==3: m = m.unsqueeze(1)
    if s.dim()==3: s = s.unsqueeze(1)
    m = (m > 0).float()
    s = (s > 0).float()
    inter = (m * s).sum().item()
    union = ((m + s) > 0).float().sum().item()
    return float(inter / max(1.0, union))

# -----------------------------
# dataset-level eval helpers
# -----------------------------

@torch.no_grad()
def accuracy(model: nn.Module, loader, device) -> float:
    model.eval(); tot=0; corr=0
    for b in loader:
        x,y,_,_,_ = _unpack_batch(b)
        x,y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        corr += (p == y).sum().item(); tot += x.size(0)
    return corr / max(tot,1)

@torch.no_grad()
def robust_accuracy(model: nn.Module, loader, device, adv_fn: Callable[[nn.Module,torch.Tensor,torch.Tensor],torch.Tensor]) -> float:
    model.eval(); tot=0; corr=0
    for b in loader:
        x,y,_,_,_ = _unpack_batch(b)
        x,y = x.to(device), y.to(device)
        xa = adv_fn(model, x, y)
        p = (torch.sigmoid(model(xa)) > 0.5).float()
        corr += (p == y).sum().item(); tot += x.size(0)
    return corr / max(tot,1)

@torch.no_grad()
def asr_with_attack(model: nn.Module, loader, device, adv_fn: Callable[[nn.Module,torch.Tensor,torch.Tensor],torch.Tensor]) -> Dict[str, float]:
    # ASR on clean-correct
    model.eval()
    clean_correct = 0
    flipped = 0
    total = 0
    for b in loader:
        x,y,_,_,_ = _unpack_batch(b)
        x,y = x.to(device), y.to(device)
        pc = (torch.sigmoid(model(x)) > 0.5).float()
        mask = (pc == y).squeeze(1) if pc.ndim==2 else (pc == y)
        total += x.size(0)
        if mask.any():
            xc, yc = x[mask], y[mask]
            xa = adv_fn(model, xc, yc)
            pa = (torch.sigmoid(model(xa)) > 0.5).float()
            flipped += (pa != yc).sum().item()
            clean_correct += yc.numel()
    return dict(asr=(flipped / max(1, clean_correct)), clean_correct=int(clean_correct), total=int(total))

# compatible with your earlier signature
@torch.no_grad()
def asr(target_model, gen, loader, device, eps=4/255):
    # gen: x -> delta in [-1,1] after tanh, scaled by eps
    target_model.eval(); gen.eval()
    right = 0; flipped = 0; tot = 0
    epsT = torch.tensor(float(eps), device=device)
    for b in loader:
        x,y,_,_,_ = _unpack_batch(b)
        x,y = x.to(device), y.to(device)
        pc = (torch.sigmoid(target_model(x)) > 0.5).float()
        mask = (pc == y)
        if mask.any():
            xc, yc = x[mask], y[mask]
            delta = epsT * gen(xc)
            xa = (xc + delta).clamp(0,1)
            pa = (torch.sigmoid(target_model(xa)) > 0.5).float()
            flipped += (pa != yc).sum().item()
            right += yc.numel()
        tot += x.size(0)
    return (flipped / max(1, right)), dict(clean_correct=right, total=tot)

# -----------------------------
# attack curves vs eps
# -----------------------------

@torch.no_grad()
def robust_curve_eps(model: nn.Module, loader, device,
                     adv_factory: Callable[[float], Callable[[nn.Module,torch.Tensor,torch.Tensor],torch.Tensor]],
                     eps_list: List[float], limit_batches: Optional[int] = None) -> List[Dict[str,float]]:
    # adv_factory returns adv_fn(model,x,y) for a given eps
    out = []
    for eps in eps_list:
        cc = 0; flips = 0; tot = 0
        k = 0
        adv_fn = adv_factory(float(eps))
        for b in loader:
            x,y,_,_,_ = _unpack_batch(b)
            x,y = x.to(device), y.to(device)
            pc = (torch.sigmoid(model(x)) > 0.5).float()
            mask = (pc == y)
            tot += x.size(0)
            if mask.any():
                xc, yc = x[mask], y[mask]
                xa = adv_fn(model, xc, yc)
                pa = (torch.sigmoid(model(xa)) > 0.5).float()
                flips += (pa != yc).sum().item()
                cc += yc.numel()
            k += 1
            if limit_batches and k >= limit_batches: break
        out.append(dict(eps=float(eps), asr=(flips/max(1,cc)), clean_correct=int(cc), total=int(tot)))
    return out

# -----------------------------
# per-sample stats logger
# -----------------------------

@torch.no_grad()
def collect_pred_table(model: nn.Module, loader, device,
                       with_logits: bool = True,
                       with_subject: bool = True) -> Dict[str, np.ndarray]:
    # builds arrays for later analysis
    model.eval()
    rows = dict(subject=[], z=[], y=[], pred=[], prob=[], logit=[])
    for b in loader:
        x,y,seg,subject,z = _unpack_batch(b)
        x,y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pr = (prob >= 0.5).float()
        rows["y"].append(y.detach().cpu().numpy())
        rows["pred"].append(pr.detach().cpu().numpy())
        rows["prob"].append(prob.detach().cpu().numpy())
        if with_logits:
            rows["logit"].append(logits.detach().cpu().numpy())
        if with_subject and subject is not None:
            rows["subject"].append(_to_numpy(subject))
            if z is not None:
                rows["z"].append(_to_numpy(z))
    for k in list(rows.keys()):
        if len(rows[k]) == 0:
            rows[k] = np.empty((0,), dtype=np.float32)
        else:
            rows[k] = np.concatenate(rows[k], axis=0)
    return rows

# -----------------------------
# subject-level aggregation
# -----------------------------

def aggregate_subject_probs(subject_ids: np.ndarray, probs: np.ndarray, y: np.ndarray,
                            method: str = "mean", thr: float = 0.5) -> Dict[str,float]:
    s = subject_ids.astype(object)
    p = _ensure_01_probs(probs).ravel()
    gt = _to_numpy(y).astype(np.int32).ravel()
    uniq = np.unique(s)
    pred_subj = []
    y_subj = []
    for u in uniq:
        m = (s == u)
        if method == "max":
            pv = float(p[m].max()) if m.any() else 0.0
        elif method == "vote":
            pv = float((p[m] >= thr).mean()) if m.any() else 0.0
        else:
            pv = float(p[m].mean()) if m.any() else 0.0
        pred_subj.append(1 if pv >= thr else 0)
        y_subj.append(int(round(gt[m].mean())) if m.any() else 0)
    pred_subj = np.array(pred_subj)
    y_subj = np.array(y_subj)
    tn, fp, fn, tp = confusion_from_preds(pred_subj, y_subj)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return dict(acc=acc, tp=tp, tn=tn, fp=fp, fn=fn, n=len(uniq))

# -----------------------------
# timing and throughput
# -----------------------------

@torch.no_grad()
def throughput(model: nn.Module, loader, device, warmup=5, iters=50) -> Dict[str,float]:
    # rough FPS on inference
    model.eval()
    # pick one batch size
    b = next(iter(loader))
    x, y, _, _, _ = _unpack_batch(b)
    x = x.to(device)
    for _ in range(max(1,warmup)):
        _ = model(x)
    torch.cuda.synchronize() if device.type=="cuda" else None
    t0 = time.time()
    for _ in range(max(1,iters)):
        _ = model(x)
    torch.cuda.synchronize() if device.type=="cuda" else None
    dt = time.time() - t0
    bs = x.size(0)
    return dict(batch_size=int(bs), iter=iters, sec=dt, fps=(iters*bs/dt))

# -----------------------------
# binning helpers
# -----------------------------

def bin_stats(values: np.ndarray, mask: np.ndarray, edges: List[float],
              clean_corr: np.ndarray, flipped: np.ndarray) -> List[Dict[str, float]]:
    # values on clean-correct tumour slices
    v = _to_numpy(values).ravel()
    m = _to_numpy(mask).astype(bool).ravel()
    cc = _to_numpy(clean_corr).astype(bool).ravel()
    fl = _to_numpy(flipped).astype(bool).ravel()
    idx = np.digitize(v, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges)-2)
    out = []
    for i in range(len(edges)-1):
        sel = (idx == i) & m
        n_tum = int(sel.sum())
        cc_i = int((sel & cc).sum())
        fl_i = int((sel & cc & fl).sum())
        asr_i = float(fl_i / max(1, cc_i))
        out.append(dict(bin_lo=float(edges[i]), bin_hi=float(edges[i+1]),
                        n_tumour=n_tum, clean_correct=cc_i, flips=fl_i, asr=asr_i))
    return out

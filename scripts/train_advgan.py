#!/usr/bin/env python3
# AdvGAN training script (BraTS 2021, 2D slices)

from __future__ import annotations
import argparse, csv, json, math, os, re, sys, time, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

# local deps
from src.models.target import load_frozen_target
from src.models.advgan_gen import AdvGANGenerator
from src.models.advgan_disc import PatchDiscriminator
from src.eval.visualize import save_triplet_grid

# -----------------------
# utils
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class CSVLogger:
    def __init__(self, path: Path, header: List[str]):
        self.path = path
        self.header = header
        ensure_dir(path.parent)
        if not path.exists():
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
    def write(self, row: Dict):
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, "") for k in self.header])

def apply_spectral_norm(m: nn.Module):
    for name, mod in m.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            try:
                nn.utils.spectral_norm(mod)
            except Exception:
                pass

# -----------------------
# eps schedule
# -----------------------
class EpsSchedule:
    def __init__(self, spec: str):
        self.spec = spec.strip()
        self.mode = "fixed"
        self.values = []
        self.start = None
        self.end = None
        self.steps = None
        if "," in self.spec:
            self.mode = "list"
            self.values = [float(x) for x in self.spec.split(",")]
        elif self.spec.startswith("linear:"):
            # linear:start:end:steps
            m = re.match(r"linear:([^:]+):([^:]+):([^:]+)", self.spec)
            if not m:
                raise ValueError("bad eps schedule")
            self.mode = "linear"
            self.start = float(m.group(1))
            self.end = float(m.group(2))
            self.steps = int(m.group(3))
        else:
            self.mode = "fixed"
            self.values = [float(self.spec)]
    def at(self, i: int) -> float:
        if self.mode == "fixed":
            return self.values[0]
        if self.mode == "list":
            return self.values[i % len(self.values)]
        if self.mode == "linear":
            if self.steps <= 1:
                return self.end
            t = min(max(i, 0), self.steps - 1) / (self.steps - 1)
            return (1.0 - t) * self.start + t * self.end
        return float(self.spec)

# -----------------------
# image losses
# -----------------------
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # x: B,C,H,W
    dh = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
    dw = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
    return dh + dw

def gaussian_window_1d(size: int, sigma: float, device):
    ax = torch.arange(size, device=device) - (size - 1) / 2.0
    w = torch.exp(-0.5 * (ax / sigma) ** 2)
    w = w / w.sum()
    return w

def ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # x,y: B,C,H,W in [0,1]
    device = x.device
    c = x.size(1)
    w1 = gaussian_window_1d(window, sigma, device)
    w2 = (w1[:, None] @ w1[None, :]).unsqueeze(0).unsqueeze(0)  # 1,1,kh,kw
    w2 = w2.expand(c, 1, window, window)
    mu_x = F.conv2d(x, w2, padding=window // 2, groups=c)
    mu_y = F.conv2d(y, w2, padding=window // 2, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    sigma_x2 = F.conv2d(x * x, w2, padding=window // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, w2, padding=window // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, w2, padding=window // 2, groups=c) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return (1.0 - ssim_map).mean()

# -----------------------
# mask ops
# -----------------------
def to_ring(mask: torch.Tensor, width: int) -> torch.Tensor:
    # mask: B,1,H,W in {0,1}
    if width <= 0:
        return mask
    k = torch.ones(1, 1, 3, 3, device=mask.device)
    dil = mask.clone()
    ero = mask.clone()
    for _ in range(width):
        dil = (F.conv2d(dil.float(), k, padding=1) > 0).float()
        ero = (F.conv2d(ero.float(), k, padding=1) >= 9).float()
    ring = (dil - ero).clamp(0, 1)
    return ring

def build_mask(seg2d: torch.Tensor, mode: str, ring_width: int) -> torch.Tensor:
    # seg2d: B,1,H,W (labels 0,1,2,4). Any >0 is tumour region.
    if mode == "none":
        return torch.ones_like(seg2d)
    base = (seg2d > 0).float()
    if mode == "seg":
        return base
    if mode == "ring":
        return to_ring(base, ring_width)
    return torch.ones_like(seg2d)

# -----------------------
# dataset
# -----------------------
class BratsSlicesCacheAdv(Dataset):
    def __init__(self, root: str, subjects: List[str], split: str,
                 augment: bool = True, empty_thr: float = 0.001,
                 max_train_slices_per_class: Optional[int] = None,
                 seed: int = 1337, jitter: float = 0.0):
        self.root = Path(root)
        self.rows = []
        rng = np.random.default_rng(seed)
        for name in subjects:
            with np.load(self.root / f"{name}.npz") as z:
                seg = z["seg"]
                nzf = z["nz_frac"]
                D = seg.shape[2]
                pos = [int(k) for k in range(D) if (seg[..., k] > 0).any()]
                neg = [int(k) for k in range(D) if not (seg[..., k] > 0).any() and float(nzf[k]) >= empty_thr]
            if split == "train" and max_train_slices_per_class:
                if len(pos) > max_train_slices_per_class:
                    pos = rng.choice(pos, size=max_train_slices_per_class, replace=False).tolist()
                if len(neg) > max_train_slices_per_class:
                    neg = rng.choice(neg, size=max_train_slices_per_class, replace=False).tolist()
            self.rows += [(name, int(z), 1) for z in pos] + [(name, int(z), 0) for z in neg]
        self.augment = augment
        self.jitter = float(jitter)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        name, z, label = self.rows[i]
        with np.load(self.root / f"{name}.npz") as npz:
            x = npz["X"][..., z].astype(np.float32) / 65535.0  # 4,H,W
            s = npz["seg"][..., z].astype(np.float32)          # H,W
        x = torch.from_numpy(x)           # 4,H,W
        y = torch.tensor([float(label)], dtype=torch.float32)
        seg = torch.from_numpy((s > 0).astype(np.float32)).unsqueeze(0)  # 1,H,W

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
                seg = torch.flip(seg, dims=[2])
            if torch.rand(1).item() < 0.3:
                x = torch.flip(x, dims=[1])
                seg = torch.flip(seg, dims=[1])
            if self.jitter > 0:
                scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.jitter
                bias  = (torch.rand(1).item() * 2 - 1) * self.jitter
                x = (x * scale + bias).clamp(0.0, 1.0)

        return x, y, seg, name, z

# -----------------------
# GAN losses
# -----------------------
class AdvLoss:
    def __init__(self, mode: str = "hinge", label_smooth: float = 0.0, gp_lambda: float = 10.0):
        self.mode = mode
        self.label_smooth = label_smooth
        self.gp_lambda = gp_lambda
        self.bce = nn.BCEWithLogitsLoss()

    def d_loss(self, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == "bce":
            tgt_r = torch.ones_like(d_real) * (1.0 - self.label_smooth)
            tgt_f = torch.zeros_like(d_fake)
            return self.bce(d_real, tgt_r) + self.bce(d_fake, tgt_f)
        if self.mode == "hinge":
            return (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
        if self.mode == "wgan":
            return -(d_real.mean() - d_fake.mean())
        return (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())

    def g_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        if self.mode == "bce":
            tgt = torch.ones_like(d_fake)
            return self.bce(d_fake, tgt)
        if self.mode == "hinge":
            return -d_fake.mean()
        if self.mode == "wgan":
            return -d_fake.mean()
        return -d_fake.mean()

def grad_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, gp_lambda: float) -> torch.Tensor:
    # real,fake: B,4,H,W
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=real.device)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    d_inter = discriminator(inter)
    grads = torch.autograd.grad(
        outputs=d_inter.sum(),
        inputs=inter,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * gp_lambda
    return gp

# -----------------------
# training helpers
# -----------------------
@dataclass
class TrainConfig:
    # data
    data_root: str
    splits: str
    train_split: str
    val_split: str
    batch: int
    workers: int
    empty_thr: float
    max_train_slices_per_class: Optional[int]
    # attack
    eps: str
    targeted: bool
    target_label: Optional[int]
    mask_mode: str
    ring_width: int
    lam_attack: float
    lam_adv: float
    lam_pert: float
    lam_tv: float
    lam_ssim: float
    pert_norm: str
    # gan
    adv_mode: str
    label_smooth: float
    gp_lambda: float
    sn_disc: bool
    # opt
    epochs: int
    steps_per_epoch: int
    max_steps_total: Optional[int]
    lrG: float
    lrD: float
    betas: Tuple[float,float]
    wd: float
    grad_accum: int
    clip_grad: Optional[float]
    amp: bool
    # io
    out_dir: str
    sample_every: int
    save_every: int
    resume: Optional[str]
    # misc
    seed: int
    smoke: bool
    jitter: float

def make_target_labels(y: torch.Tensor, targeted: bool, target_label: Optional[int]) -> torch.Tensor:
    if targeted:
        if target_label is None:
            # flip labels
            return 1.0 - y
        return torch.full_like(y, float(target_label))
    else:
        return 1.0 - y

def bce_attack_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)

def clamp_delta_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
    return delta.clamp(-eps, eps)

def mask_apply(delta: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # delta: B,4,H,W ; m: B,1,H,W
    return delta * m

def sample_grid(x: torch.Tensor, x_adv: torch.Tensor, out_path: Path):
    save_triplet_grid(
        x.detach().cpu(),
        x_adv.detach().cpu(),
        (x_adv - x).detach().cpu(),
        out_path
    )

def evaluate_asr(target_model: nn.Module, gen: nn.Module, dl: DataLoader, device, eps: float,
                 mask_mode: str, ring_width: int, max_batches: Optional[int] = None) -> Dict[str, float]:
    target_model.eval()
    gen.eval()
    tot = 0
    right = 0
    flipped = 0
    with torch.no_grad():
        for bi, (x, y, seg, _, _) in enumerate(tqdm(dl, desc="Eval ASR", leave=False), start=1):
            x = x.to(device)
            y = y.to(device)
            seg = seg.to(device)
            m = build_mask(seg, mask_mode, ring_width)
            delta = torch.tanh(gen(x))
            delta = clamp_delta_linf(delta, eps)
            delta = mask_apply(delta, m)
            x_adv = (x + delta).clamp(0, 1)
            logits_c = target_model(x)
            pred_c = (torch.sigmoid(logits_c) > 0.5).float()
            keep = (pred_c == y).squeeze(1)
            if keep.any():
                x_c = x[keep]
                y_c = y[keep]
                seg_c = seg[keep]
                m_c = build_mask(seg_c, mask_mode, ring_width)
                d = torch.tanh(gen(x_c))
                d = clamp_delta_linf(d, eps)
                d = mask_apply(d, m_c)
                xa = (x_c + d).clamp(0, 1)
                logits_a = target_model(xa)
                pred_a = (torch.sigmoid(logits_a) > 0.5).float()
                right += y_c.numel()
                flipped += (pred_a != y_c).sum().item()
            tot += x.size(0)
            if max_batches and bi >= max_batches:
                break
    asr = flipped / max(right, 1)
    return dict(asr=asr, clean_correct=right, total=tot)

# -----------------------
# main training
# -----------------------
def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "ckpts"
    samp_dir = out_dir / "samples"
    ensure_dir(out_dir); ensure_dir(ckpt_dir); ensure_dir(samp_dir)

    # splits
    splits = json.loads(Path(cfg.splits).read_text())
    train_names = splits.get(cfg.train_split, [])
    val_names   = splits.get(cfg.val_split, [])
    if cfg.smoke:
        train_names = train_names[:min(len(train_names), 40)]
        val_names   = val_names[:min(len(val_names), 20)]

    # data
    train_ds = BratsSlicesCacheAdv(
        cfg.data_root, train_names, "train",
        augment=True, empty_thr=cfg.empty_thr,
        max_train_slices_per_class=20,
        seed=cfg.seed, jitter=cfg.jitter
    )
    val_ds = BratsSlicesCacheAdv(
        cfg.data_root, val_names, "val",
        augment=False, empty_thr=cfg.empty_thr,
        max_train_slices_per_class=None,
        seed=cfg.seed
    )
    tr = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                    num_workers=cfg.workers, pin_memory=(device.type == "cuda"))
    va = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                    num_workers=cfg.workers, pin_memory=(device.type == "cuda"))

    # models
    T = load_frozen_target(str(REPO / "results" / "baseline" / "tinycnn.pt"), device)
    G = AdvGANGenerator(in_ch=4).to(device)
    D = PatchDiscriminator(in_ch=4).to(device)
    if cfg.sn_disc:
        apply_spectral_norm(D)

    # opt
    optG = torch.optim.AdamW(G.parameters(), lr=cfg.lrG, betas=cfg.betas, weight_decay=cfg.wd)
    optD = torch.optim.AdamW(D.parameters(), lr=cfg.lrD, betas=cfg.betas, weight_decay=cfg.wd)

    scaler = GradScaler(enabled=cfg.amp)
    adv = AdvLoss(cfg.adv_mode, cfg.label_smooth, cfg.gp_lambda)
    eps_sched = EpsSchedule(cfg.eps)

    step_global = 0
    start_epoch = 1

    # resume
    if cfg.resume and Path(cfg.resume).exists():
        chk = torch.load(cfg.resume, map_location=device)
        G.load_state_dict(chk["G"])
        D.load_state_dict(chk["D"])
        optG.load_state_dict(chk["optG"])
        optD.load_state_dict(chk["optD"])
        if "scaler" in chk and cfg.amp:
            scaler.load_state_dict(chk["scaler"])
        start_epoch = chk.get("epoch", 1)
        step_global = chk.get("step", 0)
        print(f"[INFO] resumed from {cfg.resume} epoch={start_epoch} step={step_global}")

    # logs
    csv_path = out_dir / "train_log.csv"
    csv_log = CSVLogger(csv_path, [
        "step","epoch","d_loss","g_loss","attack","pert","tv","ssim","eps","asr_val"
    ])

    # loss helpers
    def perturb_reg(delta: torch.Tensor) -> torch.Tensor:
        if cfg.pert_norm == "l2":
            return (delta.view(delta.size(0), -1).pow(2).sum(dim=1).sqrt().mean())
        if cfg.pert_norm == "l1":
            return delta.abs().mean()
        if cfg.pert_norm == "linf":
            return delta.abs().amax(dim=(1,2,3)).mean()
        return delta.abs().mean()

    def attack_loss_for(x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = T(x_adv)
        tgt = make_target_labels(y, cfg.targeted, cfg.target_label)
        return bce_attack_loss(logits, tgt)

    # train loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        pbar = tqdm(tr, desc=f"Train ep {epoch}", total=cfg.steps_per_epoch if cfg.steps_per_epoch else len(tr))
        G.train(); D.train()
        it_in_epoch = 0

        for (x, y, seg, _, _ ) in pbar:
            x = x.to(device)        # B,4,H,W
            y = y.to(device)        # B,1
            seg = seg.to(device)    # B,1,H,W

            eps = eps_sched.at(step_global)
            eps_t = float(eps)

            m = build_mask(seg, cfg.mask_mode, cfg.ring_width)

            # D step
            with autocast(enabled=cfg.amp):
                delta = torch.tanh(G(x))
                delta = clamp_delta_linf(delta, eps_t)
                delta = mask_apply(delta, m)
                x_adv = (x + delta).clamp(0, 1)

                d_real = D(x)
                d_fake = D(x_adv.detach())
                d_loss = adv.d_loss(d_real, d_fake)
                if cfg.adv_mode == "wgan" and cfg.gp_lambda > 0:
                    d_loss = d_loss + grad_penalty(D, x, x_adv.detach(), cfg.gp_lambda)

            optD.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(d_loss).backward()
                scaler.unscale_(optD)
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.clip_grad)
                scaler.step(optD)
            else:
                d_loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.clip_grad)
                optD.step()

            # G step
            with autocast(enabled=cfg.amp):
                delta = torch.tanh(G(x))
                delta = clamp_delta_linf(delta, eps_t)
                delta = mask_apply(delta, m)
                x_adv = (x + delta).clamp(0, 1)

                d_fake = D(x_adv)
                g_adv = adv.g_loss(d_fake)
                g_attack = attack_loss_for(x_adv, y) * cfg.lam_attack
                g_pert = perturb_reg(delta) * cfg.lam_pert
                g_tv = tv_loss(delta) * cfg.lam_tv if cfg.lam_tv > 0 else torch.zeros((), device=device)
                g_ssim = ssim_loss(x, x_adv) * cfg.lam_ssim if cfg.lam_ssim > 0 else torch.zeros((), device=device)
                g_loss = cfg.lam_adv * g_adv + g_attack + g_pert + g_tv + g_ssim

            optG.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(g_loss).backward()
                scaler.unscale_(optG)
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.clip_grad)
                scaler.step(optG)
                scaler.update()
            else:
                g_loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.clip_grad)
                optG.step()

            it_in_epoch += 1
            step_global += 1

            pbar.set_postfix(
                d=f"{float(d_loss):.3f}",
                g=f"{float(g_loss):.3f}",
                att=f"{float(g_attack):.3f}",
                prt=f"{float(g_pert):.3f}",
                eps=f"{eps_t:.5f}"
            )

            # sample
            if cfg.sample_every and (step_global % cfg.sample_every == 0):
                with torch.no_grad():
                    xb = x[:4]
                    mb = m[:4]
                    db = torch.tanh(G(xb))
                    db = clamp_delta_linf(db, eps_t)
                    db = mask_apply(db, mb)
                    xa = (xb + db).clamp(0, 1)
                    sample_grid(xb, xa, samp_dir / f"step_{step_global:07d}.png")

            # save
            if cfg.save_every and (step_global % cfg.save_every == 0):
                ck = {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict(),
                    "epoch": epoch,
                    "step": step_global,
                    "meta": {
                        "eps": eps_t,
                        "mask": cfg.mask_mode,
                        "ring": cfg.ring_width,
                        "adv_mode": cfg.adv_mode
                    }
                }
                if cfg.amp:
                    ck["scaler"] = scaler.state_dict()
                torch.save(ck, ckpt_dir / f"advgan_{step_global:07d}.pt")

            # limit steps per epoch
            if cfg.steps_per_epoch and it_in_epoch >= cfg.steps_per_epoch:
                break
            # limit total steps
            if cfg.max_steps_total and step_global >= cfg.max_steps_total:
                break

        # end epoch eval
        with torch.no_grad():
            ev = evaluate_asr(T, G, va, device, eps_sched.at(step_global), cfg.mask_mode, cfg.ring_width,
                              max_batches=50 if cfg.smoke else None)
        row = dict(
            step=step_global, epoch=epoch,
            d_loss=float(d_loss), g_loss=float(g_loss),
            attack=float(g_attack), pert=float(g_pert),
            tv=float(g_tv), ssim=float(g_ssim),
            eps=eps_sched.at(step_global), asr_val=float(ev["asr"])
        )
        csv_log.write(row)
        torch.save({
            "G": G.state_dict(), "D": D.state_dict(),
            "optG": optG.state_dict(), "optD": optD.state_dict(),
            "epoch": epoch, "step": step_global,
            "meta": {"eps": eps_sched.at(step_global), "mask": cfg.mask_mode, "ring": cfg.ring_width}
        }, out_dir / "advgan_last.pt")
        print(f"[epoch {epoch}] asr_val={ev['asr']:.3f} step={step_global}")

        if cfg.max_steps_total and step_global >= cfg.max_steps_total:
            print("[INFO] reached max steps total")
            break

    print("[DONE] training complete")

# -----------------------
# arg parsing
# -----------------------
def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--splits", default=str(REPO / "data" / "processed" / "splits.json"))
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--empty-thr", type=float, default=0.001)
    ap.add_argument("--max-train-slices-per-class", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.05)
    # attack
    ap.add_argument("--eps", type=str, default=str(4/255))
    ap.add_argument("--targeted", action="store_true")
    ap.add_argument("--target_label", type=int, choices=[0,1], default=None)
    ap.add_argument("--mask_mode", choices=["none","seg","ring"], default="none")
    ap.add_argument("--ring_width", type=int, default=2)
    ap.add_argument("--lam_attack", type=float, default=1.0)
    ap.add_argument("--lam_adv", type=float, default=0.5)
    ap.add_argument("--lam_pert", type=float, default=10.0)
    ap.add_argument("--lam_tv", type=float, default=0.0)
    ap.add_argument("--lam_ssim", type=float, default=0.0)
    ap.add_argument("--pert_norm", choices=["l1","l2","linf"], default="l2")
    # gan
    ap.add_argument("--adv_mode", choices=["hinge","bce","wgan"], default="hinge")
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--gp_lambda", type=float, default=10.0)
    ap.add_argument("--sn_disc", action="store_true")
    # opt
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--max_steps_total", type=int, default=None)
    ap.add_argument("--lrG", type=float, default=2e-4)
    ap.add_argument("--lrD", type=float, default=4e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=None)
    ap.add_argument("--amp", action="store_true")
    # io
    ap.add_argument("--out_dir", default=str(REPO / "results" / "advgan"))
    ap.add_argument("--sample_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None)
    # misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")

    args = ap.parse_args()
    if args.smoke:
        args.steps_per_epoch = min(args.steps_per_epoch, 100)
        args.epochs = min(args.epochs, 1)
        print("[SMOKE] steps_per_epoch capped to", args.steps_per_epoch)

    cfg = TrainConfig(
        data_root=args.data_root,
        splits=args.splits,
        train_split=args.train_split,
        val_split=args.val_split,
        batch=args.batch,
        workers=args.workers,
        empty_thr=args.empty_thr,
        max_train_slices_per_class=args.max_train_slices_per_class,
        eps=args.eps,
        targeted=bool(args.targeted),
        target_label=args.target_label,
        mask_mode=args.mask_mode,
        ring_width=int(args.ring_width),
        lam_attack=float(args.lam_attack),
        lam_adv=float(args.lam_adv),
        lam_pert=float(args.lam_pert),
        lam_tv=float(args.lam_tv),
        lam_ssim=float(args.lam_ssim),
        pert_norm=args.pert_norm,
        adv_mode=args.adv_mode,
        label_smooth=float(args.label_smooth),
        gp_lambda=float(args.gp_lambda),
        sn_disc=bool(args.sn_disc),
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        max_steps_total=args.max_steps_total,
        lrG=float(args.lrG),
        lrD=float(args.lrD),
        betas=(float(args.beta1), float(args.beta2)),
        wd=float(args.wd),
        grad_accum=int(args.grad_accum),
        clip_grad=args.clip_grad,
        amp=bool(args.amp),
        out_dir=args.out_dir,
        sample_every=int(args.sample_every),
        save_every=int(args.save_every),
        resume=args.resume,
        seed=int(args.seed),
        smoke=bool(args.smoke),
        jitter=float(args.jitter),
    )
    return cfg

def main():
    cfg = parse_args()
    train(cfg)

if __name__ == "__main__":
    main()

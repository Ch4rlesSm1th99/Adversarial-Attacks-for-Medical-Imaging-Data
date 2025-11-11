#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nibabel as nib

MODS = ["flair","t1","t1ce","t2"]

# ---------- RAW (fallback) ----------
def robust_minmax(slice2d, p_lo=1, p_hi=99):
    s = slice2d.astype(np.float32, copy=False)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    nz = s != 0
    vals = s[nz] if nz.any() else s
    if not np.isfinite(vals).any(): return np.zeros_like(s, dtype=np.float32)
    lo = np.nanpercentile(vals, p_lo); hi = np.nanpercentile(vals, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        finite = vals[np.isfinite(vals)]
        if finite.size: lo, hi = np.nanmin(finite), np.nanmax(finite)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(s, dtype=np.float32)
    s = np.clip(s, lo, hi)
    out = (s - lo) / (hi - lo + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

class BratsSlicesRaw(Dataset):
    def __init__(self, root, subjects, split, target_hw=(256,256),
                 p_lo=1, p_hi=99, max_train_slices_per_class=None,
                 seed=1337, augment=False, filter_empty_neg=True, empty_thr=0.001):
        self.root = Path(root); self.split = split
        self.target_hw = target_hw; self.p_lo=p_lo; self.p_hi=p_hi
        self.augment = augment; self.filter_empty_neg = filter_empty_neg; self.empty_thr=float(empty_thr)
        self.rows = []
        rng = np.random.default_rng(seed)
        for name in subjects:
            sub = self.root / name
            seg = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
            pos = [z for z in range(seg.shape[2]) if (seg[..., z] > 0).any()]
            neg = [z for z in range(seg.shape[2]) if not (seg[..., z] > 0).any()]
            if self.filter_empty_neg and neg:
                vols = [np.nan_to_num(nib.load(str(next(sub.glob(f"*_{m}.nii.gz")))).get_fdata(dtype=np.float32),
                                      nan=0.0, posinf=0.0, neginf=0.0) for m in MODS]
                union = np.logical_or.reduce([v != 0 for v in vols])
                nz_frac = union.mean(axis=(0,1))
                neg = [z for z in neg if nz_frac[z] >= self.empty_thr]
            if split == "train" and max_train_slices_per_class:
                if len(pos) > max_train_slices_per_class:
                    pos = rng.choice(pos, size=max_train_slices_per_class, replace=False).tolist()
                if len(neg) > max_train_slices_per_class:
                    neg = rng.choice(neg, size=max_train_slices_per_class, replace=False).tolist()
            self.rows += [(name, int(z), 1) for z in pos]
            self.rows += [(name, int(z), 0) for z in neg]
        self._cache_sub = None; self._cache_vols = None

    def __len__(self): return len(self.rows)

    def _load_subject_vols(self, subdir: Path):
        if self._cache_sub == subdir: return self._cache_vols
        vols = []
        for m in MODS:
            f = next(subdir.glob(f"*_{m}.nii.gz"))
            v = nib.load(str(f)).get_fdata(dtype=np.float32)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            vols.append(v)
        self._cache_sub, self._cache_vols = subdir, vols
        return vols

    def __getitem__(self, i):
        name, z, label = self.rows[i]
        sub = self.root / name
        vols = self._load_subject_vols(sub)
        chans = [robust_minmax(v[..., z]) for v in vols]
        x = torch.from_numpy(np.stack(chans, axis=0)).float().unsqueeze(0)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False).squeeze(0)
        if self.augment and torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x.clamp_(0,1), y

# ---------- CACHE (.npz) ----------
class BratsSlicesCache(Dataset):
    def __init__(self, cache_root, subjects, split, max_train_slices_per_class=None,
                 seed=1337, augment=False, empty_thr=0.001):
        self.root = Path(cache_root); self.split = split; self.augment = augment; self.rows=[]
        rng = np.random.default_rng(seed)
        for name in subjects:
            with np.load(self.root / f"{name}.npz") as z:
                seg, nzf = z["seg"], z["nz_frac"]; D = seg.shape[2]
                pos = [int(k) for k in range(D) if (seg[..., k] > 0).any()]
                neg = [int(k) for k in range(D) if not (seg[..., k] > 0).any() and float(nzf[k]) >= empty_thr]
            if split == "train" and max_train_slices_per_class:
                if len(pos) > max_train_slices_per_class:
                    pos = rng.choice(pos, size=max_train_slices_per_class, replace=False).tolist()
                if len(neg) > max_train_slices_per_class:
                    neg = rng.choice(neg, size=max_train_slices_per_class, replace=False).tolist()
            self.rows += [(name, int(z), 1) for z in pos] + [(name, int(z), 0) for z in neg]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        name, z, label = self.rows[i]
        with np.load(self.root / f"{name}.npz") as npz:
            x = npz["X"][..., z].astype(np.float32) / 65535.0
        x = torch.from_numpy(x)
        if self.augment and torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y

# ---------- model ----------
class TinyCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),    nn.ReLU(True), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

@torch.no_grad()
def evaluate(model, loader, device, criterion, max_batches=None):
    model.eval(); n=0; corr=0; loss_sum=0.0
    for bi,(x,y) in enumerate(tqdm(loader, desc="Eval", leave=False), start=1):
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = criterion(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        corr += (preds == y).sum().item(); loss_sum += loss.item()*x.size(0); n += x.size(0)
        if max_batches and bi >= max_batches: break
    return (loss_sum / max(n,1)), (corr / max(n,1))

def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--data_root", default=str(repo/"data"/"raw"),
                    help="raw NIfTI root OR cache root if .npz files are present")
    ap.add_argument("--splits",    default=str(repo/"data"/"processed"/"splits.json"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max_train_slices_per_class", type=int, default=20)
    ap.add_argument("--empty-thr", type=float, default=0.001)
    ap.add_argument("--max-batches", type=int, default=None, help="Stop after N train batches (smoke mode)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    splits = json.loads(Path(args.splits).read_text())
    DatasetCls = BratsSlicesCache if list(Path(args.data_root).glob("*.npz")) else BratsSlicesRaw

    train_ds = DatasetCls(args.data_root, splits["train"], "train",
                          max_train_slices_per_class=args.max_train_slices_per_class,
                          augment=True, empty_thr=args.empty_thr)
    val_ds   = DatasetCls(args.data_root, splits["val"], "val",
                          max_train_slices_per_class=None, augment=False, empty_thr=args.empty_thr)

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))
    va = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))

    labels = np.array([r[2] for r in train_ds.rows])
    pos = labels.sum(); neg = len(labels) - pos
    pos_weight = torch.tensor([neg / max(pos,1)], device=device, dtype=torch.float32)
    print(f"Train slices: {len(train_ds)} (pos={int(pos)}, neg={int(neg)}) | Val slices: {len(val_ds)} | using_cache={DatasetCls is BratsSlicesCache}")

    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, args.epochs + 1):
        model.train();
        n = 0;
        corr = 0;
        loss_sum = 0.0
        pbar = tqdm(tr, desc=f"Train ep {ep}", total=len(tr))
        for step, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x);
            loss = criterion(logits, y)
            loss.backward();
            opt.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            corr += (preds == y).sum().item();
            loss_sum += loss.item() * x.size(0);
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(corr / max(n, 1)):.3f}")
            if args.max_batches and step >= args.max_batches:
                print(f"[SMOKE] stopping epoch early at {args.max_batches} batches")
                break

        tr_loss, tr_acc = (loss_sum / max(n,1)), (corr / max(n,1))
        va_loss, va_acc = evaluate(model, va, device, criterion)
        print(f"[epoch {ep}] train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")

    out = repo/"results"/"baseline"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out/"tinycnn.pt")
    print("Saved:", (out/"tinycnn.pt").resolve())

if __name__ == "__main__":
    main()

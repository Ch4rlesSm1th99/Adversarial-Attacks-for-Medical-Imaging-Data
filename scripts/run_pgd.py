#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from src.models.target import load_frozen_target
from src.eval.visualize import save_triplet_grid
from scripts.eval_attack import attack_fgsm, attack_pgd

class BratsSlicesCache(Dataset):
    def __init__(self, root, subjects, split, empty_thr=0.001):
        self.root = Path(root); self.rows=[]
        for name in subjects:
            with np.load(self.root/f"{name}.npz") as z:
                seg, nzf = z["seg"], z["nz_frac"]; D=seg.shape[2]
                pos = [int(k) for k in range(D) if (seg[...,k]>0).any()]
                neg = [int(k) for k in range(D) if not (seg[...,k]>0).any() and float(nzf[k])>=empty_thr]
            self.rows += [(name,int(z),1) for z in pos] + [(name,int(z),0) for z in neg]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        name,z,label = self.rows[i]
        with np.load(self.root/f"{name}.npz") as npz:
            x = npz["X"][..., z].astype(np.float32)/65535.0
        x = torch.from_numpy(x)
        y = torch.tensor([float(label)], dtype=torch.float32)
        return x, y

@torch.no_grad()
def accuracy_tqdm(model, loader, device, max_batches=None):
    model.eval(); tot=0; corr=0
    for bi,(x,y) in enumerate(tqdm(loader, desc="Clean", leave=False), start=1):
        x,y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > 0.5).float()
        corr += (p == y).sum().item(); tot += x.size(0)
        if max_batches and bi >= max_batches: break
    return corr / max(tot,1)

def asr(model, loader, device, make_adv, max_batches=None):
    model.eval()
    clean_correct = flipped = adv_correct = total = 0
    pbar = tqdm(loader, desc="ASR", total=len(loader))
    for bi,(x,y) in enumerate(pbar, start=1):
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_pred = (torch.sigmoid(model(x)) > 0.5).float()
        mask = (clean_pred == y)
        total += x.size(0)
        if mask.any():
            x_c, y_c = x[mask], y[mask]
            clean_correct += y_c.numel()
            x_adv = make_adv(x_c, y_c)
            with torch.no_grad():
                adv_pred = (torch.sigmoid(model(x_adv)) > 0.5).float()
                flipped += (adv_pred != y_c).sum().item()
                adv_correct += (adv_pred == y_c).sum().item()
        pbar.set_postfix(clean_correct=clean_correct, flipped=flipped)
        if max_batches and bi >= max_batches:
            break
    return dict(clean_correct=int(clean_correct), flipped=int(flipped),
                adv_correct=int(adv_correct), total=int(total),
                asr=(flipped / max(clean_correct,1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="cache root (.npz)")
    ap.add_argument("--splits", default=str(REPO/"data"/"processed"/"splits.json"))
    ap.add_argument("--baseline_ckpt", default=str(REPO/"results"/"baseline"/"tinycnn.pt"))
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)

    ap.add_argument("--attack", choices=["fgsm","pgd"], default="pgd")
    ap.add_argument("--eps", type=float, default=4/255)
    ap.add_argument("--alpha", type=float, default=1/255)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--random_start", action="store_true")
    ap.add_argument("--targeted", action="store_true")
    ap.add_argument("--target_label", type=int, choices=[0,1], default=None)
    ap.add_argument("--save_grid", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(Path(args.splits).read_text())
    subjects = splits[args.split]

    ds = BratsSlicesCache(args.data_root, subjects, args.split)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))

    T = load_frozen_target(args.baseline_ckpt, device)

    clean_acc = accuracy_tqdm(T, dl, device, max_batches=args.max_batches)
    print(f"Clean accuracy ({args.split}): {clean_acc:.3f}")

    tgt = args.target_label if args.targeted else None

    def make_adv(x, y):
        if args.attack == "fgsm":
            return attack_fgsm(x, y, T, eps=args.eps, targeted=tgt,
                               mask=None, clip_min=0.0, clip_max=1.0)
        else:
            return attack_pgd(x, y, T, eps=args.eps, alpha=args.alpha, steps=args.steps,
                               targeted=tgt, rand_start=args.random_start,
                               mask=None, clip_min=0.0, clip_max=1.0)

    stats = asr(T, dl, device, make_adv, max_batches=args.max_batches)
    print(f"ASR ({args.attack}): {stats['asr']:.3f}  | clean-correct={stats['clean_correct']} of total={stats['total']}")

    if args.save_grid:
        xb, yb = [], []
        for i in range(4):
            x,y = ds[i]
            xb.append(x.numpy()); yb.append(y.numpy())
        x = torch.from_numpy(np.stack(xb, 0)).to(device)
        y = torch.from_numpy(np.stack(yb, 0)).to(device)
        x_adv = make_adv(x, y)
        figs = REPO/"results"/"figures"; figs.mkdir(parents=True, exist_ok=True)
        save_triplet_grid(x.detach().cpu(), x_adv.detach().cpu(), (x_adv-x).detach().cpu(),
                          figs/f"{args.attack}_examples.png")
        print("Saved:", (figs/f"{args.attack}_examples.png").resolve())

if __name__ == "__main__":
    main()

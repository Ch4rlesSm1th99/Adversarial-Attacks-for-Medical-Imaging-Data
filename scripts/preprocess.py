#!/usr/bin/env python3
# BraTS 2021 preprocessing and indexing

from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from tqdm import tqdm

try:
    from skimage.transform import resize
except Exception:
    resize = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

MODS = ["flair", "t1", "t1ce", "t2"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def mm2str(n: float) -> str:
    return f"{n:.6f}"

def robust_scale01(v: np.ndarray, p_lo=1, p_hi=99) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    nz = v != 0
    vals = v[nz] if nz.any() else v
    if not np.isfinite(vals).any():
        return np.zeros_like(v, dtype=np.float32)
    lo = np.nanpercentile(vals, p_lo); hi = np.nanpercentile(vals, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        finite = vals[np.isfinite(vals)]
        if finite.size:
            lo, hi = np.nanmin(finite), np.nanmax(finite)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(v, dtype=np.float32)
    v = np.clip(v, lo, hi)
    return (v - lo) / (hi - lo + 1e-6)

def union_nz_mask(*vols: np.ndarray) -> np.ndarray:
    m = None
    for v in vols:
        cur = (np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) != 0)
        m = cur if m is None else (m | cur)
    return m if m is not None else None

def discover_subjects(raw_root: Path) -> List[Path]:
    subs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")])
    if not subs:
        raise SystemExit(f"[ERR] No subjects under {raw_root}")
    return subs

def check_subject(sub: Path):
    missing = []
    vols = []
    for m in MODS + ["seg"]:
        match = list(sub.glob(f"*_{m}.nii.gz"))
        if not match:
            missing.append(m); continue
        try:
            v = nib.load(str(match[0])).get_fdata(dtype=np.float32)
            vols.append(v)
        except Exception:
            missing.append(m)
    ok = len(missing) == 0
    shape = vols[0].shape if vols else (0,0,0)
    shapes_match = ok and all(v.shape == shape for v in vols)
    return dict(name=sub.name, ok=int(ok and shapes_match), missing=";".join(missing),
                H=shape[0], W=shape[1], D=shape[2] if vols else 0, shapes_match=int(shapes_match))

def build_splits(names: List[str], val_ratio=0.15, test_ratio=0.15, seed=1337) -> Dict[str,List[str]]:
    rng = np.random.default_rng(seed)
    names = names[:]
    rng.shuffle(names)
    n = len(names)
    nv = int(round(val_ratio * n))
    nt = int(round(test_ratio * n))
    return {
        "train": names[: n - nv - nt],
        "val":   names[n - nv - nt : n - nt],
        "test":  names[n - nt :],
    }

def stratify_by_tumour_load(subjects: List[Path], bins=5, val_ratio=0.15, test_ratio=0.15, seed=1337) -> Dict[str,List[str]]:
    rng = np.random.default_rng(seed)
    stats = []
    for sub in tqdm(subjects, desc="Stratify", leave=False):
        seg_f = list(sub.glob("*_seg.nii.gz"))[0]
        seg = nib.load(str(seg_f)).get_fdata(dtype=np.float32)
        D = seg.shape[2]
        pos_frac = float(np.mean([(seg[...,z] > 0).any() for z in range(D)]))
        stats.append((sub.name, pos_frac))
    stats.sort(key=lambda x: x[1])
    bins_list = [[] for _ in range(bins)]
    for i, (name, frac) in enumerate(stats):
        bins_list[i * bins // len(stats)].append(name)
    train, val, test = [], [], []
    for bucket in bins_list:
        rng.shuffle(bucket)
        n = len(bucket); nv = int(round(val_ratio * n)); nt = int(round(test_ratio * n))
        train += bucket[: n - nv - nt]
        val   += bucket[n - nv - nt : n - nt]
        test  += bucket[n - nt :]
    return {"train": train, "val": val, "test": test}

def write_slice_index_csv(
    root: Path, splits: Dict[str,List[str]], out_csv: Path,
    from_cache: bool, empty_thr: float, max_train_slices_per_class: Optional[int],
    seed: int
):
    rng = np.random.default_rng(seed)
    rows = []
    for split, names in splits.items():
        for name in tqdm(names, desc=f"Index {split}", leave=False):
            if from_cache:
                with np.load(root / f"{name}.npz") as z:
                    seg = z["seg"]; nzf = z["nz_frac"]; D = seg.shape[2]
                    pos = [int(k) for k in range(D) if (seg[...,k] > 0).any()]
                    neg = [int(k) for k in range(D) if not (seg[...,k] > 0).any() and float(nzf[k]) >= empty_thr]
            else:
                sub = root / name
                seg = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
                vols = [nib.load(str(next(sub.glob(f"*_{m}.nii.gz")))).get_fdata(dtype=np.float32) for m in MODS]
                D = seg.shape[2]
                union = union_nz_mask(*vols)
                pos = [z for z in range(D) if (seg[...,z] > 0).any()]
                neg = [z for z in range(D) if not (seg[...,z] > 0).any() and union[...,z].mean() >= empty_thr]
            if split == "train" and max_train_slices_per_class:
                if len(pos) > max_train_slices_per_class:
                    pos = rng.choice(pos, size=max_train_slices_per_class, replace=False).tolist()
                if len(neg) > max_train_slices_per_class:
                    neg = rng.choice(neg, size=max_train_slices_per_class, replace=False).tolist()
            for z in pos + neg:
                tumour_area = float((seg[...,z] > 0).sum()) / float(seg.shape[0]*seg.shape[1])
                rows.append((name, int(z), int(z in pos), split, mm2str(tumour_area)))
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject","z","label","split","tumour_area_frac"])
        w.writerows(rows)
    print(f"[OK] slice index -> {out_csv} rows={len(rows)}")

def _overlay_preview(flair2d: np.ndarray, seg2d: np.ndarray, out_png: Path, title: str):
    if plt is None:
        warnings.warn("matplotlib missing; skip preview")
        return
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ax.imshow(flair2d, cmap="gray", vmin=0.0, vmax=1.0)
    if seg2d.max() > 0:
        overlay = np.zeros((*seg2d.shape, 4), dtype=np.float32)
        s = seg2d.astype(np.int32)
        overlay[(s==4),0] = 1.0
        overlay[(s==2),1] = 1.0
        overlay[(s==1),2] = 1.0
        overlay[...,3] = (seg2d > 0).astype(np.float32) * 0.35
        ax.imshow(overlay)
    ax.set_title(title); ax.axis("off")
    ensure_dir(out_png.parent)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def export_previews(root: Path, subjects: List[str], out_dir: Path, from_cache: bool, n_per_subject=2):
    print(f"[INFO] previews -> {out_dir}")
    for name in tqdm(subjects, desc="Previews", leave=False):
        if from_cache:
            with np.load(root / f"{name}.npz") as z:
                X = z["X"].astype(np.float32)/65535.0
                seg = z["seg"]; D = X.shape[-1]
                take = max(1, min(n_per_subject, D))
                zs = [0, D//2][:take] if take <= 2 else list(range(0, D, max(1, D // take)))[:take]
                for z in zs:
                    _overlay_preview(X[0,...,z], seg[...,z], out_dir / f"{name}_z{z:03d}.png", f"{name} z={z}")
        else:
            sub = root / name
            flair = nib.load(str(next(sub.glob("*_flair.nii.gz")))).get_fdata(dtype=np.float32)
            seg   = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
            D = flair.shape[2]
            take = max(1, min(n_per_subject, D))
            zs = [0, D//2][:take] if take <= 2 else list(range(0, D, max(1, D // take)))[:take]
            flair01 = robust_scale01(flair)
            for z in zs:
                _overlay_preview(flair01[...,z], seg[...,z], out_dir / f"{name}_z{z:03d}.png", f"{name} z={z}")

def write_2d_shards(
    root: Path, splits: Dict[str,List[str]], out_root: Path,
    from_cache: bool, shard_size: int, to_size_hw: Tuple[int,int],
    max_slices_total: Optional[int], empty_thr: float, p_lo=1, p_hi=99, seed=1337,
    keep_ratio: Optional[float] = 1.0
):
    assert resize is not None, "scikit-image required"
    rng = np.random.default_rng(seed)
    ensure_dir(out_root)
    Ht, Wt = to_size_hw
    for split, names in splits.items():
        shard_idx = 0
        buf_x, buf_y, buf_sub, buf_z = [], [], [], []
        n_total = 0
        for name in tqdm(names, desc=f"Shards2D {split}", leave=False):
            if from_cache:
                with np.load(root / f"{name}.npz") as z:
                    X = z["X"].astype(np.float32)/65535.0
                    seg = z["seg"].astype(np.float32)
                    nzf = z["nz_frac"]
            else:
                sub = root / name
                vols = [robust_scale01(nib.load(str(next(sub.glob(f"*_{m}.nii.gz")))).get_fdata(dtype=np.float32), p_lo, p_hi) for m in MODS]
                seg  = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
                H,W,D = seg.shape
                X = np.zeros((4, Ht, Wt, D), dtype=np.float32)
                for ci, v in enumerate(vols):
                    for z in range(D):
                        X[ci,...,z] = resize(v[...,z], (Ht, Wt), order=1, mode="constant", preserve_range=True, anti_aliasing=True)
                nzf = union_nz_mask(*vols).mean(axis=(0,1))
            D = X.shape[-1]
            pos = [int(z) for z in range(D) if (seg[...,z] > 0).any()]
            neg = [int(z) for z in range(D) if not (seg[...,z] > 0).any() and float(nzf[z]) >= empty_thr]
            if keep_ratio is not None and len(neg) > 0 and len(pos) > 0:
                target_neg = int(keep_ratio * len(pos))
                if target_neg < len(neg):
                    neg = rng.choice(neg, size=target_neg, replace=False).tolist()
            for z, lab in ([(z,1) for z in pos] + [(z,0) for z in neg]):
                x = X[..., z]
                buf_x.append(x); buf_y.append([float(lab)]); buf_sub.append(name); buf_z.append(z)
                if len(buf_x) >= shard_size:
                    out_f = out_root / f"shard_{split}_{shard_idx:05d}.npz"
                    np.savez_compressed(out_f, X=np.stack(buf_x,0), y=np.array(buf_y, dtype=np.float32),
                                        subject=np.array(buf_sub), z=np.array(buf_z, dtype=np.int32))
                    shard_idx += 1; buf_x.clear(); buf_y.clear(); buf_sub.clear(); buf_z.clear()
                n_total += 1
                if max_slices_total and n_total >= max_slices_total:
                    break
            if max_slices_total and n_total >= max_slices_total:
                break
        if buf_x:
            out_f = out_root / f"shard_{split}_{shard_idx:05d}.npz"
            np.savez_compressed(out_f, X=np.stack(buf_x,0), y=np.array(buf_y, dtype=np.float32),
                                subject=np.array(buf_sub), z=np.array(buf_z, dtype=np.int32))
        print(f"[OK] 2D shards {split} -> {out_root} approx_slices={n_total}")

def write_25d_shards(
    root: Path, splits: Dict[str,List[str]], out_root: Path,
    from_cache: bool, shard_size: int, to_size_hw: Tuple[int,int],
    context: int, max_slices_total: Optional[int], empty_thr: float,
    p_lo=1, p_hi=99, seed=1337
):
    assert resize is not None, "scikit-image required"
    ensure_dir(out_root)
    Ht, Wt = to_size_hw
    for split, names in splits.items():
        shard_idx = 0
        buf_x, buf_y, buf_sub, buf_z = [], [], [], []
        n_total = 0
        for name in tqdm(names, desc=f"Shards25D k={context} {split}", leave=False):
            if from_cache:
                with np.load(root / f"{name}.npz") as z:
                    X = z["X"].astype(np.float32)/65535.0
                    seg = z["seg"].astype(np.float32)
                    nzf = z["nz_frac"]
            else:
                sub = root / name
                vols = [robust_scale01(nib.load(str(next(sub.glob(f"*_{m}.nii.gz")))).get_fdata(dtype=np.float32), p_lo, p_hi) for m in MODS]
                seg  = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
                H,W,D = seg.shape
                X = np.zeros((4, Ht, Wt, D), dtype=np.float32)
                for ci, v in enumerate(vols):
                    for z in range(D):
                        X[ci,...,z] = resize(v[...,z], (Ht, Wt), order=1, mode="constant", preserve_range=True, anti_aliasing=True)
                nzf = union_nz_mask(*vols).mean(axis=(0,1))
            D = X.shape[-1]
            pos = [int(z) for z in range(D) if (seg[...,z] > 0).any()]
            neg = [int(z) for z in range(D) if not (seg[...,z] > 0).any() and float(nzf[z]) >= empty_thr]
            for z, lab in ([(z,1) for z in pos] + [(z,0) for z in neg]):
                idxs = [min(max(zz, 0), D-1) for zz in range(z-context, z+context+1)]
                stack = np.concatenate([X[:, ..., zz] for zz in idxs], axis=0)
                buf_x.append(stack); buf_y.append([float(lab)]); buf_sub.append(name); buf_z.append(z)
                if len(buf_x) >= shard_size:
                    out_f = out_root / f"shard25d_{split}_{shard_idx:05d}.npz"
                    np.savez_compressed(out_f, X=np.stack(buf_x,0), y=np.array(buf_y, dtype=np.float32),
                                        subject=np.array(buf_sub), z=np.array(buf_z, dtype=np.int32),
                                        context=np.int32(context))
                    shard_idx += 1; buf_x.clear(); buf_y.clear(); buf_sub.clear(); buf_z.clear()
                n_total += 1
                if max_slices_total and n_total >= max_slices_total:
                    break
            if max_slices_total and n_total >= max_slices_total:
                break
        if buf_x:
            out_f = out_root / f"shard25d_{split}_{shard_idx:05d}.npz"
            np.savez_compressed(out_f, X=np.stack(buf_x,0), y=np.array(buf_y, dtype=np.float32),
                                subject=np.array(buf_sub), z=np.array(buf_z, dtype=np.int32),
                                context=np.int32(context))
        print(f"[OK] 2.5D shards {split} -> {out_root} k={context} approx_slices={n_total}")

def write_3d_cache(raw_root: Path, splits: Dict[str,List[str]], out_root: Path, vol_size: Tuple[int,int,int], p_lo=1, p_hi=99):
    assert resize is not None, "scikit-image required"
    ensure_dir(out_root)
    Dz, Ht, Wt = vol_size
    subjects = sum(splits.values(), [])
    for name in tqdm(subjects, desc="3D cache", leave=False):
        sub = raw_root / name
        vols = [robust_scale01(nib.load(str(next(sub.glob(f"*_{m}.nii.gz")))).get_fdata(dtype=np.float32), p_lo, p_hi) for m in MODS]
        seg  = nib.load(str(next(sub.glob("*_seg.nii.gz")))).get_fdata(dtype=np.float32)
        X = np.zeros((4, Dz, Ht, Wt), dtype=np.float32)
        for ci, v in enumerate(vols):
            v_resz = resize(v, (Ht, Wt, Dz), order=1, mode="constant", preserve_range=True, anti_aliasing=True)
            X[ci] = np.moveaxis(v_resz, -1, 0)
        seg_resz = resize(seg, (Ht, Wt, Dz), order=0, mode="edge", preserve_range=True, anti_aliasing=False).astype(np.uint8)
        seg_resz = np.moveaxis(seg_resz, -1, 0)
        out_f = out_root / f"{name}.npz"
        np.savez_compressed(out_f, X=X, seg=seg_resz)
    print(f"[OK] 3D cache -> {out_root}")

def compute_dataset_stats_from_cache(cache_root: Path, splits: Dict[str,List[str]], out_json: Path):
    stats = {}
    for split, names in splits.items():
        n_subjects = len(names); n_slices = 0; pos_slices = 0
        for name in tqdm(names, desc=f"Stats {split}", leave=False):
            with np.load(cache_root / f"{name}.npz") as z:
                seg = z["seg"]; D = seg.shape[2]
                n_slices += D
                pos_slices += sum(int((seg[...,z] > 0).any()) for z in range(D))
        stats[split] = dict(subjects=n_subjects, slices=n_slices, pos_slices=pos_slices, pos_frac=(pos_slices/max(n_slices,1)))
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"[OK] stats -> {out_json}")

def write_dataset_yaml(out_yaml: Path, data_root_for_training: str, size_hw: Tuple[int,int], context: int):
    cfg = dict(
        data_root=data_root_for_training,
        target_size_hw=list(size_hw),
        context=int(context),
        modalities=MODS
    )
    ensure_dir(out_yaml.parent)
    out_yaml.write_text(json.dumps(cfg, indent=2))
    print(f"[OK] dataset.yaml -> {out_yaml}")

def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(repo / "data" / "raw"))
    ap.add_argument("--out_dir",   default=str(repo / "data" / "processed"))

    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--stratify", choices=["none","tumour_load"], default="none")

    ap.add_argument("--index-slices", action="store_true")
    ap.add_argument("--max-train-slices-per-class", type=int, default=20)
    ap.add_argument("--empty-thr", type=float, default=0.001)

    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--to-size", nargs=2, type=int, default=(256,256))

    ap.add_argument("--make-previews", action="store_true")
    ap.add_argument("--n-previews", type=int, default=2)

    ap.add_argument("--build-2d-shards", action="store_true")
    ap.add_argument("--build-25d-shards", action="store_true")
    ap.add_argument("--context", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--max-slices-total", type=int, default=None)

    ap.add_argument("--build-3d-cache", action="store_true")
    ap.add_argument("--vol-size", nargs=3, type=int, default=(128,256,256))

    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--auto-cache-detect", action="store_true")

    ap.add_argument("--write-stats", action="store_true")
    ap.add_argument("--write-dataset-yaml", action="store_true")

    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-subjects", type=int, default=None)
    args = ap.parse_args()

    raw_root = Path(args.data_root)
    out_dir  = Path(args.out_dir); ensure_dir(out_dir)

    if args.smoke:
        args.limit_subjects = args.limit_subjects or 20
        args.n_previews = min(args.n_previews, 2)
        args.shard_size = min(args.shard_size, 1024)
        args.max_slices_total = args.max_slices_total or 1000
        print(f"[SMOKE] subjects<={args.limit_subjects} max_slices_total={args.max_slices_total} shard_size={args.shard_size} previews={args.n_previews}")

    if args.auto_cache_detect and any(raw_root.glob("*.npz")):
        args.from_cache = True
        print("[INFO] auto cache detect")

    if args.from_cache:
        subs = sorted([p.stem for p in raw_root.glob("BraTS2021_*.npz")])
        if not subs: raise SystemExit(f"[ERR] No cache files in {raw_root}")
        if args.limit_subjects:
            subs = subs[:args.limit_subjects]
            print(f"[SMOKE] limiting discovered subjects to {len(subs)}")
        subject_names = subs
        qc_csv = out_dir / "qc_report.csv"
        qc_csv.write_text("subject,ok,missing,H,W,D,shapes_match\n")
    else:
        subject_paths = discover_subjects(raw_root)
        if args.limit_subjects:
            subject_paths = subject_paths[:args.limit_subjects]
            print(f"[SMOKE] limiting discovered subjects to {len(subject_paths)}")
        qc_rows = []
        for sub in tqdm(subject_paths, desc="QC", leave=False):
            qc = check_subject(sub)
            qc_rows.append([qc["name"], qc["ok"], qc["missing"], qc["H"], qc["W"], qc["D"], qc["shapes_match"]])
        qc_csv = out_dir / "qc_report.csv"
        with qc_csv.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject","ok","missing","H","W","D","shapes_match"]); w.writerows(qc_rows)
        print(f"[OK] QC -> {qc_csv}")
        subject_names = [p.name for p in subject_paths]

    if args.stratify == "none":
        splits = build_splits(subject_names, args.val, args.test, args.seed)
    else:
        assert not args.from_cache, "stratify requires raw NIfTI"
        splits = stratify_by_tumour_load([raw_root/p for p in subject_names], bins=5, val_ratio=args.val, test_ratio=args.test, seed=args.seed)
    splits_path = out_dir / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2))
    print(f"[OK] splits -> {splits_path} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    if args.index_slices:
        idx_csv = out_dir / "slices.csv"
        write_slice_index_csv(raw_root, splits, idx_csv, args.from_cache, args.empty_thr, args.max_train_slices_per_class, args.seed)

    if args.make_previews:
        names = (splits["train"][:2] + splits["val"][:1] + splits["test"][:1]) if not args.smoke else (splits["train"][:1] + splits["val"][:1])
        export_previews(raw_root, names, out_dir / "previews", from_cache=args.from_cache, n_per_subject=args.n_previews)

    if args.build_2d_shards:
        write_2d_shards(raw_root, splits, out_dir / "shards_2d", args.from_cache, int(args.shard_size), tuple(args.to_size),
                        args.max_slices_total, args.empty_thr, args.p_lo, args.p_hi, args.seed, keep_ratio=1.0)

    if args.build_25d_shards:
        write_25d_shards(raw_root, splits, out_dir / f"shards_25d_k{args.context}", args.from_cache, int(args.shard_size), tuple(args.to_size),
                         int(args.context), args.max_slices_total, args.empty_thr, args.p_lo, args.p_hi, args.seed)

    if args.build_3d_cache:
        assert not args.from_cache, "3D cache expects raw NIfTI"
        write_3d_cache(raw_root, splits, out_dir / f"volcache_{args.vol_size[0]}x{args.vol_size[1]}x{args.vol_size[2]}",
                       tuple(args.vol_size), args.p_lo, args.p_hi)

    if args.write_stats and args.from_cache:
        compute_dataset_stats_from_cache(raw_root, splits, out_dir / "stats_from_cache.json")

    if args.write_dataset_yaml:
        data_root_training = str(raw_root)
        write_dataset_yaml(out_dir / "dataset.yaml", data_root_training, tuple(args.to_size),
                           context=(args.context if args.build_25d_shards else 0))

    print("[DONE] preprocess complete.")

if __name__ == "__main__":
    main()

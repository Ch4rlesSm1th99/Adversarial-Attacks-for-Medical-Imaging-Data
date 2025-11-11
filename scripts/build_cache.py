#!/usr/bin/env python3
"""
Build a fast per-subject cache for BraTS:
- Loads 4 modalities + seg from raw NIfTI
- Robust scales each modality to [0,1] (p1..p99 on non-zero voxels)
- Resizes in-plane to 256x256
- Saves X:[4,H,W,D] as uint16 (0..65535), seg:[H,W,D] as uint8, nz_frac:[D] as float32
Outputs: <out_root>/<subject>.npz
"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    from skimage.transform import resize
except ImportError as e:
    raise SystemExit("Please install scikit-image: pip install scikit-image") from e

MODS = ["flair","t1","t1ce","t2"]

def robust_scale(v: np.ndarray, p_lo=1, p_hi=99):
    v = v.astype(np.float32, copy=False)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    m = v != 0
    vals = v[m] if m.any() else v
    if not np.isfinite(vals).any(): return np.zeros_like(v, dtype=np.float32)
    lo = np.nanpercentile(vals, p_lo); hi = np.nanpercentile(vals, p_hi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        finite = vals[np.isfinite(vals)]
        if finite.size: lo, hi = np.nanmin(finite), np.nanmax(finite)
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            return np.zeros_like(v, dtype=np.float32)
    v = np.clip(v, lo, hi)
    out = (v - lo) / (hi - lo + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

def load_vol(subdir: Path, mod: str) -> np.ndarray:
    f = next(subdir.glob(f"*_{mod}.nii.gz"))
    return nib.load(str(f)).get_fdata(dtype=np.float32)

def load_seg(subdir: Path) -> np.ndarray:
    f = next(subdir.glob(f"*_seg.nii.gz"))
    return nib.load(str(f)).get_fdata(dtype=np.float32)

def to_uint16(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    return (x * 65535.0 + 0.5).astype(np.uint16)

def process_subject(subdir: Path, out_root: Path, p_lo: float, p_hi: float, target_hw: tuple[int,int]) -> str:
    name = subdir.name
    out_f = out_root / f"{name}.npz"
    if out_f.exists(): return f"[skip] {name}"
    vols = [load_vol(subdir, m) for m in MODS]
    seg  = load_seg(subdir).astype(np.uint8, copy=False)
    H, W, D = seg.shape
    scaled = [robust_scale(v, p_lo, p_hi) for v in vols]

    th, tw = target_hw
    scaled_res = []
    for v in scaled:
        out = np.zeros((th, tw, D), dtype=np.float32)
        for z in range(D):
            out[..., z] = resize(v[..., z], (th, tw), order=1, mode="constant",
                                 preserve_range=True, anti_aliasing=True)
        scaled_res.append(out)
    seg_res = np.zeros((th, tw, D), dtype=np.uint8)
    for z in range(D):
        seg_res[..., z] = resize(seg[..., z], (th, tw), order=0, mode="edge",
                                 preserve_range=True, anti_aliasing=False).astype(np.uint8)

    union_res = np.logical_or.reduce([s != 0 for s in scaled_res])
    nz_frac = union_res.mean(axis=(0,1)).astype(np.float32)

    X = np.stack(scaled_res, axis=0)  # [4, th, tw, D]
    X16 = to_uint16(X)
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_f, X=X16, seg=seg_res, nz_frac=nz_frac, mods=np.array(MODS))
    return f"[ok] {name}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="WSL path containing BraTS2021_0xxxxx")
    ap.add_argument("--out_root", required=True, help="Where to write .npz cache")
    ap.add_argument("--clip", nargs=2, type=float, default=(1.0, 99.0))
    ap.add_argument("--to-size", nargs=2, type=int, default=(256, 256))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit-subjects", type=int, default=None)
    args = ap.parse_args()

    raw = Path(args.raw_root); out = Path(args.out_root)
    subs = sorted([p for p in raw.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")])
    if not subs: raise SystemExit(f"No subjects found under {raw}")
    if args.limit_subjects:
        subs = subs[:args.limit_subjects]
        print(f"[SMOKE] limiting subjects to {len(subs)}")

    print(f"Subjects: {len(subs)}  | out: {out}")
    futs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for s in subs:
            futs.append(ex.submit(process_subject, s, out, args.clip[0], args.clip[1], tuple(args.to_size)))
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Cache"):
            _ = _.result()
    print("Cache build complete.")

if __name__ == "__main__":
    main()

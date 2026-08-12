"""Detector-only evaluation on density-counting datasets (Mall, Beijing-BRT).

For each frame: count person detections and compare to GT head-point count.
Reports MAE, MSE, and per-frame absolute error distribution.

Mall GT: scipy.io loads mall_gt.mat -> 'count' (Nx1 int) and 'frame' (Nx2 cell with head xy).
BRT GT: scipy.io loads .mat -> typically 'image_info' or 'annPoints' with head xy points.

Usage:
    python tools/detector_eval.py --dataset mall --frames-dir datasets/mall/mall_dataset/frames \
        --gt-mat datasets/mall/mall_dataset/mall_gt.mat --model yolov8n_crowdhuman.pt \
        --sample 200 --out mall_results/detector.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
from ultralytics import YOLO


def load_mall_gt(mat_path: Path) -> list[int]:
    """Return per-frame person count list, indexed 0..N-1."""
    m = sio.loadmat(str(mat_path))
    if "count" in m:
        c = m["count"].flatten()
        return [int(x) for x in c]
    if "frame" in m:
        # MAT cell array: each frame[i] has a struct with 'loc' Nx2
        frames = m["frame"][0]
        return [len(f[0][0][0]) if len(f) else 0 for f in frames]
    raise SystemExit(f"unknown Mall GT format: keys={list(m.keys())}")


def load_brt_gt(gt_dir: Path) -> dict[str, int]:
    """BRT: one .mat per image with 'image_info' or 'annPoints'. Return {name: count}."""
    counts = {}
    for mat in gt_dir.glob("*.mat"):
        m = sio.loadmat(str(mat))
        # try common keys
        for key in ("annPoints", "image_info", "points", "loc"):
            if key in m:
                arr = m[key]
                if hasattr(arr, "shape") and arr.ndim >= 2:
                    counts[mat.stem] = int(arr.shape[0])
                    break
        else:
            counts[mat.stem] = 0
    return counts


def detect_count(model: YOLO, img_path: Path, imgsz: int, conf: float) -> int:
    res = model.predict(str(img_path), imgsz=imgsz, conf=conf, classes=[0], verbose=False)
    if not res:
        return 0
    boxes = res[0].boxes
    if boxes is None:
        return 0
    return int(len(boxes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mall", "brt"], required=True)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--gt-mat", type=Path, help="Mall: single mall_gt.mat")
    ap.add_argument("--gt-dir", type=Path, help="BRT: ground_truth folder of .mat files")
    ap.add_argument("--model", default="yolov8n_crowdhuman.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=200,
                    help="random sample of frames to evaluate (Mall has 2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    print(f"loading model {args.model}")
    model = YOLO(args.model)

    if args.dataset == "mall":
        if not args.gt_mat:
            raise SystemExit("--gt-mat required for Mall")
        gt_counts = load_mall_gt(args.gt_mat)
        all_imgs = sorted(args.frames_dir.glob("seq_*.jpg"))
        if len(all_imgs) != len(gt_counts):
            print(f"warn: frames={len(all_imgs)} vs gt={len(gt_counts)}")
        items = list(zip(all_imgs, gt_counts))
    else:
        if not args.gt_dir:
            raise SystemExit("--gt-dir required for BRT")
        gt_map = load_brt_gt(args.gt_dir)
        items = []
        for img in sorted(args.frames_dir.glob("*.jpg")):
            if img.stem in gt_map:
                items.append((img, gt_map[img.stem]))

    print(f"total frames: {len(items)}")
    if args.sample and len(items) > args.sample:
        random.seed(args.seed)
        items = random.sample(items, args.sample)
        print(f"sampled: {len(items)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    abs_errors = []
    for i, (img, gt) in enumerate(items, 1):
        pred = detect_count(model, img, args.imgsz, args.conf)
        err = abs(pred - gt)
        rows.append((img.name, gt, pred, err))
        abs_errors.append(err)
        if i % 25 == 0:
            print(f"  {i}/{len(items)}  MAE so far={np.mean(abs_errors):.2f}")

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(np.square(abs_errors)))
    rmse = float(np.sqrt(mse))
    median_err = float(np.median(abs_errors))

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt", "pred", "abs_err"])
        for r in rows:
            w.writerow(r)

    print(f"\n=== {args.dataset.upper()} detector eval ===")
    print(f"frames: {len(items)}")
    print(f"MAE:    {mae:.2f}")
    print(f"RMSE:   {rmse:.2f}")
    print(f"median abs err: {median_err:.1f}")
    avg_gt = float(np.mean([gt for _, gt, _, _ in rows]))
    avg_pred = float(np.mean([p for _, _, p, _ in rows]))
    print(f"avg GT/pred: {avg_gt:.1f} / {avg_pred:.1f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

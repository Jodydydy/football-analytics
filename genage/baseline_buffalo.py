"""
Baseline: pretrained InsightFace buffalo_l (genderage) on the SAME val split
as our trained MobileNetV3 model, so we can answer "was training worth it?".

buffalo_l runs SCRFD detection internally. Our val images are already tight
aligned face crops, so we pad a reflect border before inference to give the
detector some context; images where no face is detected are reported as a
separate detection-failure bucket (NOT counted as gender/age errors), and
metrics are computed on the detected subset.

Gender mapping: manifest 0=male,1=female ; buffalo_l f.sex 'M'->0 'F'->1.
Age: buffalo_l f.age (int years) -> our 9 fine buckets and the 3 coarse groups
(0-19 / 20-49 / 50+, the recommended business segmentation).

Usage:
  python baseline_buffalo.py [--limit N] [--det-size 320]
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))

# fine bucket boundaries (upper-inclusive): index by age years
def age_to_fine(a):
    if a <= 2:  return 0
    if a <= 9:  return 1
    if a <= 19: return 2
    if a <= 29: return 3
    if a <= 39: return 4
    if a <= 49: return 5
    if a <= 59: return 6
    if a <= 69: return 7
    return 8

# 3 coarse groups: 0-19 -> 0, 20-49 -> 1, 50+ -> 2
def fine_to_coarse(b):
    if b <= 2:  return 0
    if b <= 5:  return 1
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--pad", type=float, default=0.3, help="reflect border frac")
    ap.add_argument("--limit", type=int, default=0, help="0=all val rows")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                rows.append((r["path"], int(r["gender"]),
                             int(r["age_bucket"]), r["source"]))
    if args.limit:
        rows = rows[:args.limit]

    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda
                 else ["CPUExecutionProvider"])
    ctx_id = 0 if has_cuda else -1
    print(f"providers={providers}  rows={len(rows)}")
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    n = len(rows)
    n_det = 0
    g_ok = 0
    fine_ok = 0
    coarse_ok = 0
    # also track per-source on detected subset
    src_stat = {}
    t0 = time.time()
    for i, (path, g, ab, src) in enumerate(rows):
        img = cv2.imread(path)
        if img is None:
            continue
        if args.pad > 0:
            ph, pw = int(img.shape[0] * args.pad), int(img.shape[1] * args.pad)
            img = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT)
        faces = app.get(img)
        if not faces:
            continue
        # largest face
        f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
        n_det += 1
        gp = 0 if f.sex == "M" else 1
        pf = age_to_fine(int(f.age))
        g_hit = int(gp == g)
        f_hit = int(pf == ab)
        c_hit = int(fine_to_coarse(pf) == fine_to_coarse(ab))
        g_ok += g_hit; fine_ok += f_hit; coarse_ok += c_hit
        s = src_stat.setdefault(src, [0, 0, 0, 0])  # det, g, fine, coarse
        s[0] += 1; s[1] += g_hit; s[2] += f_hit; s[3] += c_hit
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  det={n_det}  {dt:.0f}s  "
                  f"({(i+1)/dt:.1f} img/s)")

    print(f"\n=== buffalo_l baseline ===  (det_size={args.det_size}, pad={args.pad})")
    print(f"detection rate: {n_det}/{n} = {n_det/n:.4f}")
    if n_det:
        print(f"on detected subset (n={n_det}):")
        print(f"  gender:        {g_ok/n_det:.4f}")
        print(f"  age 9-bucket:  {fine_ok/n_det:.4f}")
        print(f"  age 3-group:   {coarse_ok/n_det:.4f}  (0-19/20-49/50+)")
        print("\nper-source (detected subset):")
        print(f"{'src':<10} {'det':>6} {'gender':>8} {'age9':>7} {'age3':>7}")
        for src, s in src_stat.items():
            d = s[0] or 1
            print(f"{src:<10} {s[0]:>6} {s[1]/d:>8.4f} {s[2]/d:>7.4f} {s[3]/d:>7.4f}")


if __name__ == "__main__":
    main()

"""
Re-score the trained checkpoint on COARSE age groups (business-facing).

The 9 fine buckets are collapsed into 4 groups whose boundaries fall on
existing bucket edges, so no label noise is introduced:

    child   (0-9)   -> fine {0,1}        0-2, 3-9
    young   (10-29) -> fine {2,3}        10-19, 20-29
    adult   (30-49) -> fine {4,5}        30-39, 40-49
    senior  (50+)   -> fine {6,7,8}      50-59, 60-69, 70+

A prediction is correct if its fine bucket lands in the same coarse group as
the true fine bucket. Gender is unchanged (reported again for convenience).

Usage:
  python eval_coarse.py --ckpt runs/mnv3_v1/best.pt
"""
import argparse
import csv
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GenAgeNet, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))

# Several candidate business segmentations. Each maps the 9 fine buckets
# (0:0-2 1:3-9 2:10-19 3:20-29 4:30-39 5:40-49 6:50-59 7:60-69 8:70+) into
# coarse groups. We report accuracy for every scheme so the analyst can pick
# the segmentation the model can actually deliver at high accuracy.
SCHEMES = {
    "A 4grp 0-9/10-29/30-49/50+": (
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3},
        ["0-9", "10-29", "30-49", "50+"]),
    "B 4grp 0-19/20-39/40-59/60+": (
        {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3},
        ["0-19", "20-39", "40-59", "60+"]),
    "C 3grp child/adult/senior 0-19/20-49/50+": (
        {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2},
        ["0-19", "20-49", "50+"]),
    "D 2grp young/older <=29 / 30+": (
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
        ["0-29", "30+"]),
}


class EvalDS(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, g, a, src = self.rows[i]
        return self.tf(Image.open(path).convert("RGB")), g, a, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "mnv3_v1", "best.pt"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                rows.append((r["path"], int(r["gender"]),
                             int(r["age_bucket"]), r["source"]))
    print(f"val rows: {len(rows)}  device={device}")

    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ld = DataLoader(EvalDS(rows, tf), batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=device == "cuda")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GenAgeNet(n_age=9, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # per-scheme accumulators + global gender + collect true/pred fine arrays
    g_ok = 0
    n_tot = 0
    sch_ok = {name: 0 for name in SCHEMES}
    confs = {name: [[0] * len(names) for _ in range(len(names))]
             for name, (_, names) in SCHEMES.items()}

    with torch.no_grad():
        for x, g, a, idx in ld:
            x = x.to(device, non_blocking=True)
            gl, al = model(x)
            gp = gl.argmax(1).cpu()
            ap_ = al.argmax(1).cpu()
            for k in range(g.numel()):
                g_ok += int(gp[k] == g[k])
                n_tot += 1
                ta, pa = int(a[k]), int(ap_[k])
                for name, (m, _) in SCHEMES.items():
                    ct, cp = m[ta], m[pa]
                    sch_ok[name] += int(cp == ct)
                    confs[name][ct][cp] += 1

    print(f"\nGender accuracy: {g_ok/n_tot:.4f}   (n={n_tot})")
    print("\n=== Age accuracy by business segmentation ===")
    print(f"{'scheme':<42} {'groups':>7} {'accuracy':>9}")
    for name, (_, names) in SCHEMES.items():
        print(f"{name:<42} {len(names):>7} {sch_ok[name]/n_tot:>9.4f}")

    for name, (_, names) in SCHEMES.items():
        print(f"\n--- confusion: {name} (rows=true, cols=pred) ---")
        print("            " + " ".join(f"{n[:7]:>8}" for n in names))
        for i in range(len(names)):
            row = " ".join(f"{confs[name][i][j]:>8}" for j in range(len(names)))
            tot = sum(confs[name][i]) or 1
            print(f"{names[i]:<11} {row}   (rec={confs[name][i][i]/tot:.3f})")


if __name__ == "__main__":
    main()

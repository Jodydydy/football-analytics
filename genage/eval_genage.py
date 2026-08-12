"""
Detailed evaluation of a trained gender+age checkpoint on the manifest val split.

Reports:
  - gender accuracy (overall + per source)
  - age exact / adjacent-1 accuracy / MAE in buckets (overall + per source)
  - age confusion matrix (9x9)

Usage:
  python eval_genage.py --ckpt runs/mnv3_v1/best.pt
"""
import argparse
import csv
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GenAgeNet, AGE_BUCKETS, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))


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

    # accumulators
    stat = defaultdict(lambda: dict(g_ok=0, a_ok=0, a_adj=0, a_mae=0.0, n=0))
    conf = [[0] * 9 for _ in range(9)]

    with torch.no_grad():
        for x, g, a, idx in ld:
            x = x.to(device, non_blocking=True)
            gl, al = model(x)
            gp = gl.argmax(1).cpu()
            ap_ = al.argmax(1).cpu()
            for k in range(g.numel()):
                src = rows[idx[k]][3]
                for bucket in ("all", src):
                    s = stat[bucket]
                    s["g_ok"] += int(gp[k] == g[k])
                    s["a_ok"] += int(ap_[k] == a[k])
                    s["a_adj"] += int(abs(int(ap_[k]) - int(a[k])) <= 1)
                    s["a_mae"] += abs(int(ap_[k]) - int(a[k]))
                    s["n"] += 1
                conf[int(a[k])][int(ap_[k])] += 1

    print("\n=== Metrics ===")
    print(f"{'subset':<10} {'n':>7} {'gender':>8} {'age_exact':>10} "
          f"{'age_adj1':>9} {'age_mae':>8}")
    for k in ("all", "fairface", "utkface"):
        s = stat[k]
        if not s["n"]:
            continue
        print(f"{k:<10} {s['n']:>7} {s['g_ok']/s['n']:>8.4f} "
              f"{s['a_ok']/s['n']:>10.4f} {s['a_adj']/s['n']:>9.4f} "
              f"{s['a_mae']/s['n']:>8.3f}")

    print("\n=== Age confusion (rows=true, cols=pred) ===")
    hdr = "      " + " ".join(f"{i:>5}" for i in range(9))
    print(hdr)
    for i in range(9):
        row = " ".join(f"{conf[i][j]:>5}" for j in range(9))
        tot = sum(conf[i]) or 1
        print(f"{i} {AGE_BUCKETS[i]:>5} {row}   (rec={conf[i][i]/tot:.2f})")


if __name__ == "__main__":
    main()

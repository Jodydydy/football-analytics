"""
Quick free win check: test-time augmentation (horizontal flip) on the current
checkpoint. Averages softmax over original + hflip, reports gender + age
(9-bucket exact and 3-group 0-19/20-49/50+) with vs without TTA.
No retraining.

Usage:
  python eval_tta.py --ckpt runs/mnv3_v1/best.pt
"""
import argparse, csv, os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GenAgeNet, IMAGENET_MEAN, IMAGENET_STD

HERE = os.path.dirname(os.path.abspath(__file__))
def coarse(b): return 0 if b <= 2 else (1 if b <= 5 else 2)


class DS(Dataset):
    def __init__(self, rows, tf): self.rows, self.tf = rows, tf
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, g, a, s = self.rows[i]
        return self.tf(Image.open(p).convert("RGB")), g, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "mnv3_v1", "best.pt"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                rows.append((r["path"], int(r["gender"]), int(r["age_bucket"]), r["source"]))
    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ld = DataLoader(DS(rows, tf), batch_size=args.batch, shuffle=False,
                    num_workers=0, pin_memory=device == "cuda")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GenAgeNet(n_age=9, pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    acc = {k: dict(g=0, a9=0, a3=0) for k in ("plain", "tta")}
    n = 0
    with torch.no_grad():
        for x, g, a in ld:
            x = x.to(device)
            gl, al = model(x)
            glf, alf = model(torch.flip(x, dims=[3]))   # hflip
            gp_p, ap_p = F.softmax(gl, 1), F.softmax(al, 1)
            gp_t = (gp_p + F.softmax(glf, 1)) / 2
            ap_t = (ap_p + F.softmax(alf, 1)) / 2
            for name, gp, ap in (("plain", gp_p, ap_p), ("tta", gp_t, ap_t)):
                pg = gp.argmax(1).cpu(); pa = ap.argmax(1).cpu()
                acc[name]["g"] += int((pg == g).sum())
                acc[name]["a9"] += int((pa == a).sum())
                acc[name]["a3"] += int(sum(coarse(int(pa[k])) == coarse(int(a[k]))
                                           for k in range(a.numel())))
            n += a.numel()

    print(f"val n={n}")
    print(f"{'variant':<8} {'gender':>8} {'age9':>8} {'age3':>8}")
    for k in ("plain", "tta"):
        s = acc[k]
        print(f"{k:<8} {s['g']/n:>8.4f} {s['a9']/n:>8.4f} {s['a3']/n:>8.4f}")


if __name__ == "__main__":
    main()

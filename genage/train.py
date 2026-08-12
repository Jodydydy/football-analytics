"""
Multi-task gender + age training on the unified FairFace+UTKFace manifest.

Backbone: torchvision MobileNetV3-Large (ImageNet pretrained), two linear heads:
  - gender: 2-way (0=male, 1=female)
  - age:    9-way (FairFace buckets 0-2 .. 70+)

Images are already aligned face crops -> just resize + light aug.

Usage:
  python train.py --epochs 8 --batch 128
  python train.py --limit 2000 --epochs 1        # quick smoke test
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
AGE_BUCKETS = ["0-2", "3-9", "10-19", "20-29", "30-39",
               "40-49", "50-59", "60-69", "70+"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GenAgeDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, gender, age, _src, _split = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), gender, age


def load_rows(manifest, split, limit=0):
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            rows.append((r["path"], int(r["gender"]), int(r["age_bucket"]),
                         r["source"], r["split"]))
    if limit:
        rows = rows[:limit]
    return rows


class GenAgeNet(nn.Module):
    def __init__(self, n_age=9, pretrained=True):
        super().__init__()
        m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        self.features = m.features
        self.avgpool = m.avgpool
        in_feat = m.classifier[0].in_features  # 960
        self.shared = nn.Sequential(
            nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.2))
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, n_age)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        return self.gender_head(x), self.age_head(x)


def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    g_correct = g_total = 0
    a_exact = a_adj = a_total = 0
    a_mae = 0.0
    # per-source breakdown
    by_src = {}
    for x, g, a in loader:
        x = x.to(device, non_blocking=True)
        gl, al = model(x)
        gp = gl.argmax(1).cpu()
        ap = al.argmax(1).cpu()
        g_correct += (gp == g).sum().item()
        g_total += g.numel()
        a_exact += (ap == a).sum().item()
        a_adj += ((ap - a).abs() <= 1).sum().item()
        a_mae += (ap - a).abs().sum().item()
        a_total += a.numel()
    return {
        "gender_acc": g_correct / max(g_total, 1),
        "age_exact_acc": a_exact / max(a_total, 1),
        "age_adj1_acc": a_adj / max(a_total, 1),
        "age_mae_buckets": a_mae / max(a_total, 1),
        "n": g_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs", "mnv3_v1"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--age-weight", type=float, default=1.0,
                    help="weight of age loss vs gender loss")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")

    train_rows = load_rows(args.manifest, "train", args.limit)
    val_rows = load_rows(args.manifest, "val", args.limit)
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    train_tf, val_tf = make_transforms(args.size)
    train_ds = GenAgeDataset(train_rows, train_tf)
    val_ds = GenAgeDataset(val_rows, val_tf)
    pin = device == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    # age class weights (inverse freq) to counter imbalance
    age_counts = Counter(r[2] for r in train_rows)
    n_age = 9
    total = sum(age_counts.values())
    age_w = torch.tensor(
        [total / (n_age * max(age_counts.get(i, 0), 1)) for i in range(n_age)],
        dtype=torch.float32, device=device)
    print("age class weights:", [round(w, 2) for w in age_w.tolist()])

    model = GenAgeNet(n_age=n_age, pretrained=not args.no_pretrained).to(device)
    gender_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    age_crit = nn.CrossEntropyLoss(weight=age_w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(len(train_ld) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for it, (x, g, a) in enumerate(train_ld):
            x = x.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                gl, al = model(x)
                loss = gender_crit(gl, g) + args.age_weight * age_crit(al, a)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            if it % 100 == 0:
                print(f"  e{epoch} it{it}/{len(train_ld)} "
                      f"loss={loss.item():.3f} lr={sched.get_last_lr()[0]:.2e}")
        m = evaluate(model, val_ld, device)
        dt = time.time() - t0
        # combined score: prioritise age (gender is ~solved) but keep both
        score = m["gender_acc"] + m["age_exact_acc"] + 0.5 * m["age_adj1_acc"]
        m.update(epoch=epoch, train_loss=run_loss / max(len(train_ld), 1),
                 sec=round(dt, 1))
        history.append(m)
        print(f"[epoch {epoch}] {dt:.0f}s  "
              f"gender={m['gender_acc']:.4f}  age_exact={m['age_exact_acc']:.4f} "
              f"age_adj1={m['age_adj1_acc']:.4f}  age_mae={m['age_mae_buckets']:.3f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "metrics": m,
                        "age_buckets": AGE_BUCKETS, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best (score={score:.4f})")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("done. best score:", best)


if __name__ == "__main__":
    main()

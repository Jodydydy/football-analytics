"""
train_v2 — push age 3-group accuracy (0.851 -> target >=0.90) WITHOUT changing
the edge backbone (MobileNetV3-Large stays). Bundle:

  1. DLDL soft age labels  : Gaussian label distribution (sigma) + KL loss
     instead of hard CE. Age errors are almost all adjacent (+-1 = 0.94), so an
     ordinal-aware target converts near-misses into correct coarse-group hits.
  2. Aux 3-group head      : extra 512->3 head trained on 0-19/20-49/50+ (CE).
     Predicting coarse directly is easier than 9-way -> collapse.
  3. Balanced sampler      : per-sample weight ~ (1/bucket_freq)^p to boost the
     scarce/hard 50+ group without overfitting tiny 70+ (softened, p=0.5).
  4. 256 input resolution  : age cues are fine texture.
  5. 18 epochs + EMA + OneCycle.

Keeps the 9-bucket age head (ONNX/infer compatibility) AND adds the 3-group
head. Eval reports both: age3 collapsed from 9-head and age3 from the new head.

Usage:
  python train_v2.py --out-dir runs/mnv3_v2 --epochs 18 --size 256
"""
import argparse, copy, csv, json, os, time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from train import (AGE_BUCKETS, IMAGENET_MEAN, IMAGENET_STD, load_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
N_AGE = 9


def coarse(b):
    return 0 if b <= 2 else (1 if b <= 5 else 2)


class GenAgeNet2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        self.features = m.features
        self.avgpool = m.avgpool
        in_feat = m.classifier[0].in_features  # 960
        self.shared = nn.Sequential(
            nn.Linear(in_feat, 512), nn.Hardswish(), nn.Dropout(0.2))
        self.gender_head = nn.Linear(512, 2)
        self.age_head = nn.Linear(512, N_AGE)     # kept for ONNX compatibility
        self.age3_head = nn.Linear(512, 3)        # new business-group head

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        return self.gender_head(x), self.age_head(x), self.age3_head(x)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(),
                        model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


class DS(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, g, a, _src, _split = self.rows[i]
        return self.tf(Image.open(path).convert("RGB")), g, a, coarse(a)


def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size + 20, size + 20)),
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


def soft_age_targets(sigma):
    """[9,9] matrix; row a = Gaussian label distribution centred at bucket a."""
    idx = torch.arange(N_AGE).float()
    M = torch.empty(N_AGE, N_AGE)
    for a in range(N_AGE):
        d = -((idx - a) ** 2) / (2 * sigma ** 2)
        M[a] = torch.softmax(d, dim=0)
    return M


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    g_ok = a9_ok = adj = a3c_ok = a3h_ok = n = 0
    # per 3-group recall from the 3-head
    grp_tot = [0, 0, 0]; grp_ok = [0, 0, 0]
    for x, g, a, c in loader:
        x = x.to(device, non_blocking=True)
        gl, al, a3 = model(x)
        pg = gl.argmax(1).cpu(); pa = al.argmax(1).cpu(); p3 = a3.argmax(1).cpu()
        g_ok += int((pg == g).sum())
        a9_ok += int((pa == a).sum())
        adj += int((pa - a).abs().le(1).sum())
        for k in range(a.numel()):
            ca = coarse(int(a[k]))
            a3c_ok += int(coarse(int(pa[k])) == ca)
            a3h_ok += int(int(p3[k]) == ca)
            grp_tot[ca] += 1
            grp_ok[ca] += int(int(p3[k]) == ca)
        n += a.numel()
    return {
        "gender_acc": g_ok / n,
        "age9_exact": a9_ok / n,
        "age_adj1": adj / n,
        "age3_from9": a3c_ok / n,       # collapse 9-head -> 3 groups
        "age3_head": a3h_ok / n,        # dedicated 3-head
        "grp_recall": [round(grp_ok[i] / max(grp_tot[i], 1), 3) for i in range(3)],
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs", "mnv3_v2"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=1.0, help="DLDL gaussian sigma")
    ap.add_argument("--sampler-p", type=float, default=0.5,
                    help="balanced sampler strength: weight ~ (1/freq)^p")
    ap.add_argument("--age-weight", type=float, default=1.0)
    ap.add_argument("--age3-weight", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}  size={args.size}")

    train_rows = load_rows(args.manifest, "train", args.limit)
    val_rows = load_rows(args.manifest, "val", args.limit)
    print(f"train={len(train_rows)}  val={len(val_rows)}")

    train_tf, val_tf = make_transforms(args.size)
    train_ds = DS(train_rows, train_tf)
    val_ds = DS(val_rows, val_tf)

    # softened class-balanced sampler over age buckets (boost scarce 50+)
    counts = Counter(r[2] for r in train_rows)
    bucket_w = {b: (1.0 / counts[b]) ** args.sampler_p for b in counts}
    sample_w = torch.tensor([bucket_w[r[2]] for r in train_rows],
                            dtype=torch.double)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(train_rows),
                                    replacement=True)
    # effective (resampled) bucket share, for logging
    eff = {b: bucket_w[b] * counts[b] for b in counts}
    tot_eff = sum(eff.values())
    print("orig 50+ share:",
          round(sum(counts[b] for b in (6, 7, 8)) / len(train_rows), 3),
          " resampled 50+ share:",
          round(sum(eff[b] for b in (6, 7, 8)) / tot_eff, 3))

    pin = device == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=pin,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    model = GenAgeNet2(pretrained=not args.no_pretrained).to(device)
    ema = ModelEMA(model, decay=args.ema_decay)
    SOFT = soft_age_targets(args.sigma).to(device)        # [9,9]
    gender_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    age3_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
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
        run = 0.0
        for it, (x, g, a, c) in enumerate(train_ld):
            x = x.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            soft = SOFT[a]                       # [B,9] target distribution
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                gl, al, a3 = model(x)
                loss_g = gender_crit(gl, g)
                loss_a = F.kl_div(F.log_softmax(al, dim=1), soft,
                                  reduction="batchmean")
                loss_a3 = age3_crit(a3, c)
                loss = (loss_g + args.age_weight * loss_a
                        + args.age3_weight * loss_a3)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run += loss.item()
            if it % 150 == 0:
                print(f"  e{epoch} it{it}/{len(train_ld)} loss={loss.item():.3f} "
                      f"(g={loss_g.item():.2f} a={loss_a.item():.3f} "
                      f"a3={loss_a3.item():.2f}) lr={sched.get_last_lr()[0]:.2e}")
        m = evaluate(ema.ema, val_ld, device)     # eval EMA weights
        dt = time.time() - t0
        score = m["age3_head"]                    # optimise the business metric
        m.update(epoch=epoch, train_loss=run / max(len(train_ld), 1),
                 sec=round(dt, 1))
        history.append(m)
        print(f"[epoch {epoch}] {dt:.0f}s  gender={m['gender_acc']:.4f}  "
              f"age9={m['age9_exact']:.4f}  adj1={m['age_adj1']:.4f}  "
              f"age3_from9={m['age3_from9']:.4f}  age3_head={m['age3_head']:.4f}  "
              f"grp_rec={m['grp_recall']}")
        if score > best:
            best = score
            torch.save({"model": ema.ema.state_dict(), "metrics": m,
                        "age_buckets": AGE_BUCKETS, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))
            print(f"  -> saved best (age3_head={score:.4f})")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best age3_head={best:.4f}")


if __name__ == "__main__":
    main()

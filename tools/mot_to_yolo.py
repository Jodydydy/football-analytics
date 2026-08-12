"""Convert MOT17 gt.txt → YOLO-format per-image labels for fine-tuning.

Reads gt.txt for each train sequence, splits into first-half / second-half by
frame number (matching annotations/train_half.json semantics — first 50% of
frames per sequence), and emits one .txt label file per .jpg in the split.

Output layout (compatible with ultralytics YOLO data.yaml):
    out_root/
      images/
        train/<seq>/<frame>.jpg     # symlink-or-copy from images/half/<seq>-half/img1/
        val/<seq>/<frame>.jpg
      labels/
        train/<seq>/<frame>.txt     # YOLO format: class cx cy w h normalized
        val/<seq>/<frame>.txt

YOLO label line: `<class> <cx> <cy> <w> <h>` all in [0,1].
We use a single class (0 = person) for our pipeline (`main.py` requests
classes=[0] from ultralytics anyway).

Usage:
    python tools/mot_to_yolo.py \
        --mot-root datasets/mot17/MOT17 \
        --out datasets/mot17_yolo
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

# Same filter as mot_to_crossing_gt: only walking pedestrians, visible enough.
PEDESTRIAN_CLASSES = {1}


def parse_seqinfo(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def parse_gt(gt_path: Path, min_vis: float = 0.1) -> dict[int, list[tuple[float, float, float, float]]]:
    """frame → list of (cx, cy, w, h) in PIXELS, only valid pedestrians."""
    per_frame: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with gt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame = int(parts[0])
            bl, bt, bw, bh = (float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]))
            conf = float(parts[6])
            cls = int(float(parts[7]))
            vis = float(parts[8])
            if conf < 1.0 or cls not in PEDESTRIAN_CLASSES or vis < min_vis:
                continue
            cx = bl + bw / 2.0
            cy = bt + bh / 2.0
            per_frame[frame].append((cx, cy, bw, bh))
    return dict(per_frame)


def write_yolo_labels(per_frame, n_frames: int, w: int, h: int,
                       img_dir: Path, label_dir: Path):
    """Emit one .txt per frame, even if no detections (empty file)."""
    label_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(1, n_frames + 1):
        boxes = per_frame.get(frame, [])
        lbl = label_dir / f"{frame:06d}.txt"
        with lbl.open("w") as f:
            for cx, cy, bw, bh in boxes:
                # Normalize to [0,1]; clip to be safe.
                ncx = max(0.0, min(1.0, cx / w))
                ncy = max(0.0, min(1.0, cy / h))
                nw = max(0.0, min(1.0, bw / w))
                nh = max(0.0, min(1.0, bh / h))
                f.write(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", required=True, type=Path,
                    help="MOT17 root containing images/train/<seq>-SDP/")
    ap.add_argument("--out", required=True, type=Path,
                    help="output root for YOLO-format dataset")
    ap.add_argument("--min-vis", type=float, default=0.1)
    ap.add_argument("--symlink", action="store_true",
                    help="symlink images instead of copying (saves disk)")
    args = ap.parse_args()

    train_seqs_dir = args.mot_root / "images" / "train"
    if not train_seqs_dir.exists():
        raise SystemExit(f"missing {train_seqs_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    seq_stats = []
    for seq_dir in sorted(train_seqs_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        seqinfo = parse_seqinfo(seq_dir / "seqinfo.ini")
        n_frames = int(seqinfo.get("seqLength", "0"))
        w = int(seqinfo.get("imWidth", "0"))
        h = int(seqinfo.get("imHeight", "0"))
        if not n_frames or not w or not h:
            print(f"  SKIP {seq_dir.name}: bad seqinfo")
            continue
        gt = seq_dir / "gt" / "gt.txt"
        if not gt.exists():
            print(f"  SKIP {seq_dir.name}: no gt.txt")
            continue
        per_frame = parse_gt(gt, min_vis=args.min_vis)
        n_train = n_frames // 2
        # First half → train, second half → val (same convention as
        # MOT17 train_half.json / val_half.json).

        for split, frame_lo, frame_hi in [
            ("train", 1, n_train),
            ("val", n_train + 1, n_frames),
        ]:
            seq_imgs_out = args.out / "images" / split / seq_dir.name
            seq_lbls_out = args.out / "labels" / split / seq_dir.name
            seq_imgs_out.mkdir(parents=True, exist_ok=True)
            seq_lbls_out.mkdir(parents=True, exist_ok=True)

            for frame in range(frame_lo, frame_hi + 1):
                src_img = seq_dir / "img1" / f"{frame:06d}.jpg"
                dst_img = seq_imgs_out / f"{frame:06d}.jpg"
                if not src_img.exists():
                    continue
                if not dst_img.exists():
                    if args.symlink:
                        try:
                            dst_img.symlink_to(src_img.resolve())
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_img, dst_img)
                    else:
                        shutil.copy2(src_img, dst_img)
                # write label
                boxes = per_frame.get(frame, [])
                lbl = seq_lbls_out / f"{frame:06d}.txt"
                with lbl.open("w") as f:
                    for cx, cy, bw, bh in boxes:
                        ncx = max(0.0, min(1.0, cx / w))
                        ncy = max(0.0, min(1.0, cy / h))
                        nw = max(0.0, min(1.0, bw / w))
                        nh = max(0.0, min(1.0, bh / h))
                        f.write(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
        seq_stats.append((seq_dir.name, n_frames, n_train,
                          sum(len(b) for b in per_frame.values())))

    print(f"\n{'seq':<24} frames  train/val  total_boxes")
    for s, n, t, b in seq_stats:
        print(f"{s:<24} {n:>6}  {t}/{n-t:<6}  {b}")

    # data.yaml
    yaml_path = args.out / "mot17.yaml"
    yaml_path.write_text(f"""# YOLO data config for MOT17 fine-tune (1-class person)
path: {args.out.resolve().as_posix()}
train: images/train
val: images/val
nc: 1
names: ['person']
""")
    print(f"\nwrote {yaml_path}")


if __name__ == "__main__":
    main()

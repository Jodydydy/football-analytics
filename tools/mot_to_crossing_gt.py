"""Convert MOT17/MOT20 gt.txt to events.csv (frame,event) for line-crossing eval.

MOT17 gt.txt format (per-row):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis

Where:
- conf in {0,1}: 1 = active in evaluation, 0 = ignored region/static
- class: 1 = pedestrian (we keep these), 7 = static person (skip)
        2 = vehicle, 3 = bicycle, 6 = sitting, 12 = obstruction (skip)
- vis: visibility 0..1

We build per-track-id trajectory of bottom-center anchor (matches our pipeline's
default --anchor bottom-center), then for a virtual line emit one event per
sign-flip crossing. IN/OUT polarity follows the same sv.LineZone convention
the rest of the pipeline uses (signed cross-product), so the SAME --start/--end
fed to main.py and to this tool will give consistent labels.

Usage:
    python tools/mot_to_crossing_gt.py \
        --gt datasets/mot17/train/MOT17-04-DPM/gt/gt.txt \
        --start 960,0 --end 960,1080 \
        --out datasets/mot17/MOT17-04_gt.csv

Tip: pick the line with pick_line.py on the FIRST frame from img1/ (e.g.
img1/000001.jpg). Use the same --start/--end when running main.py against the
sequence.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_xy(s: str) -> tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def side_of_line(px: float, py: float,
                 ax: float, ay: float,
                 bx: float, by: float) -> float:
    """Signed cross-product. >0 left of line A→B, <0 right, =0 on the line."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


# Class IDs we treat as 'person' for line-crossing.
# MOT17 ground truth distinguishes:
#   1 = walking pedestrian        (KEEP — primary signal)
#   2 = riding pedestrian         (skip — bike, may cross too fast/multi-anchor)
#   3 = car / 4 = bicycle / 5 = motorbike / 6 = non-motorized vehicle  (skip)
#   7 = static person             (skip — sitting/standing, no crossing)
#   8 = distractor / 9 = occluder (skip — annotation noise)
#   10 = occluder on the ground   (skip)
#   11 = occluder full / 12 = reflection (skip)
PEDESTRIAN_CLASSES = {1}


def parse_gt(gt_path: Path,
             min_visibility: float = 0.0) -> dict[int, list[tuple[int, float, float]]]:
    """track_id → sorted list of (frame, anchor_x, anchor_y).

    Anchor = bottom-center of bbox (matches main.py --anchor bottom-center).
    Filters: conf=1 only, class in PEDESTRIAN_CLASSES, vis >= min_visibility.
    """
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    n_rows = n_kept = 0
    n_skip_class: dict[int, int] = defaultdict(int)

    with gt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            n_rows += 1
            frame = int(parts[0])
            tid = int(parts[1])
            bl, bt, bw, bh = (float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]))
            conf = float(parts[6])
            cls = int(float(parts[7]))
            vis = float(parts[8])
            if conf < 1.0:
                continue
            if cls not in PEDESTRIAN_CLASSES:
                n_skip_class[cls] += 1
                continue
            if vis < min_visibility:
                continue
            ax = bl + bw / 2.0
            ay = bt + bh        # bottom-center
            tracks[tid].append((frame, ax, ay))
            n_kept += 1

    for tid in tracks:
        tracks[tid].sort()

    return dict(tracks), n_rows, n_kept, dict(n_skip_class)


def emit_events(tracks: dict[int, list[tuple[int, float, float]]],
                ax: int, ay: int, bx: int, by: int,
                invert: bool) -> list[tuple[int, str, int]]:
    """For each track, emit one event per sign-flip across the line.

    Same convention as tools/caviar_gt.py:
    positive→negative side flip = 'in', negative→positive = 'out'.
    Use --invert to swap if pipeline polarity is opposite.

    Returns: list of (frame, event, track_id).
    """
    events: list[tuple[int, str, int]] = []
    in_label, out_label = ("out", "in") if invert else ("in", "out")

    for tid, traj in tracks.items():
        prev_side = None
        for (fnum, x, y) in traj:
            s = side_of_line(x, y, ax, ay, bx, by)
            if abs(s) < 1e-6:
                continue
            cur_side = 1 if s > 0 else -1
            if prev_side is not None and cur_side != prev_side:
                if prev_side > 0 and cur_side < 0:
                    events.append((fnum, in_label, tid))
                else:
                    events.append((fnum, out_label, tid))
            prev_side = cur_side

    events.sort()
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path,
                    help="MOT17 gt/gt.txt file")
    ap.add_argument("--start", required=True, type=parse_xy,
                    help="line start x,y (pixels in MOT image coords)")
    ap.add_argument("--end", required=True, type=parse_xy,
                    help="line end x,y (pixels in MOT image coords)")
    ap.add_argument("--invert", action="store_true",
                    help="swap in↔out (use if pipeline emits opposite polarity)")
    ap.add_argument("--min-visibility", type=float, default=0.1,
                    help="drop GT entries with vis < this (default 0.1; "
                         "MOT marks heavily-occluded as low vis)")
    ap.add_argument("--out", required=True, type=Path, help="output CSV")
    args = ap.parse_args()

    tracks, n_rows, n_kept, skip_class = parse_gt(args.gt,
                                                  min_visibility=args.min_visibility)
    ax, ay = args.start
    bx, by = args.end
    events = emit_events(tracks, ax, ay, bx, by, args.invert)

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "event"])
        for fnum, ev, _tid in events:
            w.writerow([fnum, ev])

    n_in = sum(1 for _, e, _ in events if e == "in")
    n_out = sum(1 for _, e, _ in events if e == "out")
    print(f"{args.gt}:")
    print(f"  rows: {n_rows} total, {n_kept} kept (class=1 pedestrian, "
          f"vis>={args.min_visibility})")
    if skip_class:
        skipped = ", ".join(f"cls{k}={v}" for k, v in sorted(skip_class.items()))
        print(f"  skipped non-pedestrian: {skipped}")
    print(f"  tracks: {len(tracks)}, events: {len(events)} "
          f"({n_in} in, {n_out} out)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

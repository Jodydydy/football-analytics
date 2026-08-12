"""Auto-derive a counter-zone polygon for each CAFE cafe from GT queueing boxes.

For each cafe in CAFE Dataset, collect ALL boxes with activity=1 (Queueing)
across all its clips, compute the bounding rectangle of those box anchors
(bottom-center, matching main.py default), expand by `--padding` percent, and
emit a 4-corner polygon JSON compatible with pick_zone.py.

This is the autonomous version of `pick_zone.py` for F3 queue-alert: instead
of operator clicking, we use the GT to learn where queueing actually happens
in each cafe's view — then the pipeline only counts persons whose anchor falls
inside that zone.

Usage:
    python tools/auto_pick_zone.py \
        --cafes 6,7,17,18,19,20 \
        --out-dir cafe_zones/ \
        --padding 0.15

Output:  cafe_zones/cafe_<id>.json with {"polygon": [[x,y], ...]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# tools/ is one level under the repo root; cafe_helper.py lives there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cafe_helper import DEFAULT_ROOT, gt_for_clip, build_index


def collect_queueing_anchors(cafe_id: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    """Return Nx2 array of (anchor_x, anchor_y) for every queueing box across
    all clips in this cafe. Anchor = bottom-center of bbox.
    """
    idx = build_index(root)
    pts: list[tuple[float, float]] = []
    for key in idx.keys():
        c_id_str, clip_id_str = key.split("/")
        if int(c_id_str) != cafe_id:
            continue
        clip_id = int(clip_id_str)
        # only fetch for clips that actually have class-1 boxes
        if "1" not in idx[key].get("classes", {}):
            continue
        rows = gt_for_clip(cafe_id, clip_id, root)
        for r in rows:
            if r["activity"] != 1:
                continue
            ax = (r["x1"] + r["x2"]) / 2.0
            ay = r["y2"]                   # bottom-center
            pts.append((ax, ay))
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


def auto_polygon(anchors: np.ndarray, padding: float = 0.15,
                 width: int = 0, height: int = 0,
                 percentile: float = 5.0) -> list[list[int]]:
    """Percentile-based bbox with `padding` ratio expansion, clipped to image.

    Uses [percentile..100-percentile] to drop outlier queueing anchors
    (e.g. occluded queue tail off-frame, or one-off persons standing in
    a non-typical spot). With percentile=5 we keep the central 90% of points.

    Returns 4 corners CW: [TL, TR, BR, BL].
    """
    if len(anchors) == 0:
        raise ValueError("no anchors")
    x_min = float(np.percentile(anchors[:, 0], percentile))
    x_max = float(np.percentile(anchors[:, 0], 100 - percentile))
    y_min = float(np.percentile(anchors[:, 1], percentile))
    y_max = float(np.percentile(anchors[:, 1], 100 - percentile))
    w = x_max - x_min
    h = y_max - y_min
    px = w * padding
    py = h * padding
    x_min -= px; y_min -= py
    x_max += px; y_max += py
    if width > 0:
        x_min = max(0, x_min); x_max = min(width, x_max)
    if height > 0:
        y_min = max(0, y_min); y_max = min(height, y_max)
    return [
        [int(round(x_min)), int(round(y_min))],
        [int(round(x_max)), int(round(y_min))],
        [int(round(x_max)), int(round(y_max))],
        [int(round(x_min)), int(round(y_max))],
    ]


def detect_frame_size(cafe_id: int, root: Path = DEFAULT_ROOT) -> tuple[int, int]:
    """Read first available clip's first frame to get (width, height)."""
    import cv2
    from cafe_helper import clip_dir, clip_frames
    idx = build_index(root)
    for key in idx.keys():
        c_id_str, clip_id_str = key.split("/")
        if int(c_id_str) != cafe_id:
            continue
        frames = clip_frames(cafe_id, int(clip_id_str), root=root)
        if frames:
            img = cv2.imread(str(frames[0]))
            if img is not None:
                return img.shape[1], img.shape[0]
    return 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cafes", default="",
                    help="comma-separated cafe ids; empty = all cafes with class-1 boxes")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--padding", type=float, default=0.15,
                    help="expand bbox by this ratio on each side (default 15%%)")
    ap.add_argument("--percentile", type=float, default=5.0,
                    help="drop outlier anchors below this percentile / above (100-percentile). "
                         "Default 5 keeps central 90%% of points.")
    ap.add_argument("--width", type=int, default=0,
                    help="frame width to clip polygon to (0=auto-detect from first frame)")
    ap.add_argument("--height", type=int, default=0,
                    help="frame height to clip polygon to (0=auto-detect)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.cafes:
        cafe_ids = [int(s) for s in args.cafes.split(",") if s]
    else:
        # All cafes that contain at least one queueing box.
        idx = build_index()
        seen = set()
        for k, v in idx.items():
            if "1" in v.get("classes", {}):
                seen.add(int(k.split("/")[0]))
        cafe_ids = sorted(seen)

    print(f"# cafes to process: {len(cafe_ids)}")
    summary = []
    for cid in cafe_ids:
        anchors = collect_queueing_anchors(cid)
        if len(anchors) == 0:
            print(f"  cafe {cid}: no queueing anchors — skip")
            continue
        # auto-detect frame size unless explicitly given
        w, h = args.width, args.height
        if w == 0 or h == 0:
            w_det, h_det = detect_frame_size(cid)
            if w_det and h_det:
                w, h = w or w_det, h or h_det
        try:
            poly = auto_polygon(anchors, args.padding, w, h,
                                percentile=args.percentile)
        except ValueError as e:
            print(f"  cafe {cid}: {e}")
            continue
        out_path = args.out_dir / f"cafe_{cid}.json"
        out_path.write_text(json.dumps({"polygon": poly}, indent=2))
        x_min, y_min = poly[0]
        x_max, y_max = poly[2]
        bx, by = (x_min + x_max) // 2, (y_min + y_max) // 2
        print(f"  cafe {cid:>2}: {len(anchors):>5} anchors -> "
              f"box ({x_min},{y_min})-({x_max},{y_max})  "
              f"size {x_max-x_min}x{y_max-y_min}  centroid ({bx},{by})  "
              f"-> {out_path.name}")
        summary.append({"cafe": cid, "n_anchors": len(anchors),
                        "bbox": poly})

    summary_path = args.out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()

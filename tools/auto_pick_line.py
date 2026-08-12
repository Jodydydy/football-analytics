"""Find an optimal counting-line for a MOT17 sequence by analyzing GT tracks.

Approach:
1. Read gt.txt, build per-track-id trajectories of bottom-center anchors.
2. Filter to long-enough tracks (≥ MIN_LEN frames), class=1 pedestrian, vis ≥ MIN_VIS.
3. Compute each track's net displacement vector (start → end).
4. Find dominant flow direction (mean of normalized vectors weighted by length).
5. Place a counting line PERPENDICULAR to that flow, through the centroid of
   all anchors, clipped to image bounds.

This is the autonomous version of pick_line.py — no human click needed.

Usage:
    python tools/auto_pick_line.py \
        --gt datasets/mot17/MOT17/images/train/MOT17-02-SDP/gt/gt.txt \
        --width 1920 --height 1080
    # prints recommended --start X,Y --end X,Y
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_gt(gt_path: Path, min_vis: float = 0.1) -> dict[int, list[tuple[int, float, float]]]:
    """Same as mot_to_crossing_gt: track_id → sorted list of (frame, x, y)."""
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with gt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            bl, bt, bw, bh = (float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]))
            conf = float(parts[6])
            cls = int(float(parts[7]))
            vis = float(parts[8])
            if conf < 1.0 or cls != 1 or vis < min_vis:
                continue
            ax = bl + bw / 2.0
            ay = bt + bh
            tracks[tid].append((frame, ax, ay))
    for tid in tracks:
        tracks[tid].sort()
    return dict(tracks)


def find_dominant_flow(tracks, min_displacement: float = 30.0,
                       min_track_len: int = 10):
    """Return (flow_dx, flow_dy) unit vector and (centroid_x, centroid_y)
    averaged over moving tracks.
    """
    vecs: list[tuple[float, float, float]] = []  # (dx, dy, length)
    all_xy: list[tuple[float, float]] = []
    for tid, traj in tracks.items():
        if len(traj) < min_track_len:
            continue
        x0, y0 = traj[0][1], traj[0][2]
        x1, y1 = traj[-1][1], traj[-1][2]
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < min_displacement:
            continue
        vecs.append((dx, dy, length))
        for _, x, y in traj:
            all_xy.append((x, y))
    if not vecs:
        return None, None, 0
    # Sum normalized × length so longer tracks vote stronger but symmetric pairs
    # don't cancel. Trick: square the direction (multiply angle by 2) so
    # opposing flows reinforce each other instead of cancelling.
    sx = sy = 0.0
    for dx, dy, L in vecs:
        nx, ny = dx / L, dy / L
        # double-angle: (cos2θ, sin2θ) = (nx²-ny², 2·nx·ny)
        sx += (nx * nx - ny * ny) * L
        sy += (2 * nx * ny) * L
    angle2 = np.arctan2(sy, sx)
    flow_angle = angle2 / 2.0
    fx, fy = float(np.cos(flow_angle)), float(np.sin(flow_angle))

    cx = float(np.mean([p[0] for p in all_xy]))
    cy = float(np.mean([p[1] for p in all_xy]))
    return (fx, fy), (cx, cy), len(vecs)


def line_perpendicular(flow, centroid, width: int, height: int):
    """Return (start, end) line perpendicular to `flow` through `centroid`,
    clipped to image bounds.
    """
    fx, fy = flow
    cx, cy = centroid
    # perpendicular = rotate 90°
    px, py = -fy, fx
    # extend to far edges
    huge = 1e6
    x1, y1 = cx + px * huge, cy + py * huge
    x2, y2 = cx - px * huge, cy - py * huge
    # Clip line to image rect via parametric: P(t) = C + t * (px, py)
    # find ts where x ∈ [0,W] and y ∈ [0,H]
    ts: list[float] = []
    if abs(px) > 1e-9:
        ts.append((0 - cx) / px)
        ts.append((width - cx) / px)
    if abs(py) > 1e-9:
        ts.append((0 - cy) / py)
        ts.append((height - cy) / py)
    # The line is unbounded; valid points have BOTH coords in range.
    valid_ts = []
    for t in ts:
        x = cx + t * px
        y = cy + t * py
        if -1 <= x <= width + 1 and -1 <= y <= height + 1:
            valid_ts.append(t)
    if len(valid_ts) < 2:
        # fallback: just clip naive
        sx, sy = max(0, min(int(round(x1)), width)), max(0, min(int(round(y1)), height))
        ex, ey = max(0, min(int(round(x2)), width)), max(0, min(int(round(y2)), height))
        return (sx, sy), (ex, ey)
    t_min, t_max = min(valid_ts), max(valid_ts)
    sx = int(round(cx + t_min * px))
    sy = int(round(cy + t_min * py))
    ex = int(round(cx + t_max * px))
    ey = int(round(cy + t_max * py))
    sx = max(0, min(sx, width))
    sy = max(0, min(sy, height))
    ex = max(0, min(ex, width))
    ey = max(0, min(ey, height))
    return (sx, sy), (ex, ey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--width", required=True, type=int)
    ap.add_argument("--height", required=True, type=int)
    ap.add_argument("--min-displacement", type=float, default=30.0,
                    help="ignore tracks with net movement < this (px)")
    ap.add_argument("--min-track-len", type=int, default=10,
                    help="ignore tracks shorter than this many frames")
    args = ap.parse_args()

    tracks = parse_gt(args.gt)
    flow, centroid, n_used = find_dominant_flow(
        tracks,
        min_displacement=args.min_displacement,
        min_track_len=args.min_track_len,
    )
    if flow is None:
        print("ERROR: no qualifying tracks found")
        return
    print(f"# {args.gt}")
    print(f"# tracks total: {len(tracks)}, qualifying (moving >={args.min_displacement}px): {n_used}")
    print(f"# dominant flow direction: ({flow[0]:+.3f}, {flow[1]:+.3f})  "
          f"angle={np.degrees(np.arctan2(flow[1], flow[0])):.1f}°")
    print(f"# centroid of anchors: ({centroid[0]:.0f}, {centroid[1]:.0f})")
    start, end = line_perpendicular(flow, centroid, args.width, args.height)
    print(f"--start {start[0]},{start[1]} --end {end[0]},{end[1]}")


if __name__ == "__main__":
    main()

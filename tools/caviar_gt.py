"""Convert CAVIAR CVML XML ground truth to events.csv (frame,event).

CAVIAR CVML format (per frame):
    <frame number="N">
      <objectlist>
        <object id="K">
          <box h="80" w="40" xc="200" yc="150"/>
          ...
        </object>
      </objectlist>
    </frame>

We build per-object-id trajectories of centroids, then for a user-chosen virtual
line emit one event per object on the frame the centroid sign flips. The
"in" direction = sign goes from + to - of the line normal (configurable via
--invert).

Usage:
    python tools/caviar_gt.py --xml fra1gt.xml \
        --start 200,0 --end 200,288 \
        --out gt.csv

Tip: pick --start/--end with pick_line.py on the first frame of the matching
.mpg, exactly the same way you do for main.py. Then run main.py with the same
--start/--end and compare with evaluate.py.
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
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


def parse_cvml(xml_path: Path) -> dict[int, list[tuple[int, float, float]]]:
    """object_id → sorted list of (frame, xc, yc)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    for frame in root.iter("frame"):
        fnum = int(frame.attrib["number"])
        for ol in frame.iter("objectlist"):
            for obj in ol.iter("object"):
                oid = int(obj.attrib["id"])
                box = obj.find("box")
                if box is None:
                    continue
                xc = float(box.attrib["xc"])
                yc = float(box.attrib["yc"])
                tracks[oid].append((fnum, xc, yc))

    for oid in tracks:
        tracks[oid].sort()
    return dict(tracks)


def emit_events(tracks: dict[int, list[tuple[int, float, float]]],
                ax: int, ay: int, bx: int, by: int,
                invert: bool) -> list[tuple[int, str]]:
    """For each track, find first sign flip across the line and emit one event.

    By default: positive→negative side flip = 'in', negative→positive = 'out'.
    Use --invert to swap.
    """
    events: list[tuple[int, str]] = []
    in_label, out_label = ("out", "in") if invert else ("in", "out")

    for oid, traj in tracks.items():
        prev_side = None
        for (fnum, xc, yc) in traj:
            s = side_of_line(xc, yc, ax, ay, bx, by)
            if abs(s) < 1e-6:
                continue  # exactly on line, skip
            cur_side = 1 if s > 0 else -1
            if prev_side is not None and cur_side != prev_side:
                # sign flipped
                if prev_side > 0 and cur_side < 0:
                    events.append((fnum, in_label))
                else:
                    events.append((fnum, out_label))
            prev_side = cur_side

    events.sort()
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, type=Path,
                    help="CAVIAR CVML XML ground truth file")
    ap.add_argument("--start", required=True, type=parse_xy, help="line start x,y")
    ap.add_argument("--end", required=True, type=parse_xy, help="line end x,y")
    ap.add_argument("--invert", action="store_true",
                    help="swap in↔out (use if pipeline emits opposite polarity)")
    ap.add_argument("--out", required=True, type=Path, help="output CSV")
    args = ap.parse_args()

    tracks = parse_cvml(args.xml)
    ax, ay = args.start
    bx, by = args.end
    events = emit_events(tracks, ax, ay, bx, by, args.invert)

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "event"])
        for fnum, ev in events:
            w.writerow([fnum, ev])

    n_in = sum(1 for _, e in events if e == "in")
    n_out = sum(1 for _, e in events if e == "out")
    print(f"{args.xml.name}: {len(tracks)} tracks -> {len(events)} events ({n_in} in, {n_out} out)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

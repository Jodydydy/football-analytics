"""Convert Collective Activity Dataset annotations -> queue event events.csv.

Annotation format (per .txt, one line per detection):
    frame_no  x  y  w  h  class_id  pose_id

class_id meanings (Choi 2009 Collective Activity):
    1 = Crossing
    2 = Waiting
    3 = Queueing       <-- our target
    4 = Walking
    5 = Talking

We emit one "queue start" event each time the number of Queueing persons
goes from 0 -> >=1 (transition into a queue), and one "queue end" event
on the reverse transition. With --min-persons N, transitions happen at
the N-threshold instead of 1.

Output CSV columns: frame, event   (event in {"queue_start","queue_end"})
which evaluate.py can consume by passing direction "queue_start" or
"queue_end" - or use --emit alerts to write a single "alert" event per
queue formation (start frame only).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

QUEUEING_CLASS = 3


def parse_seq_annotation(txt_path: Path, class_id: int) -> dict[int, int]:
    """Return {frame_no: number_of_persons_with_class_id_in_that_frame}.

    Annotation columns (CAD Augmented): det_id, frame, x, y, w, h, class_id, pose_id
    """
    counts: dict[int, int] = defaultdict(int)
    with txt_path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                frame = int(parts[1])
                cls = int(parts[6])
            except ValueError:
                continue
            if cls == class_id:
                counts[frame] += 1
    return counts


def emit_events(counts: dict[int, int], min_persons: int,
                emit: str, min_duration: int, max_gap: int) -> list[tuple[int, str]]:
    """Emit queue alert events with hysteresis.

    A queue event is a contiguous run of frames where count>=min_persons,
    longer than min_duration frames, with at most max_gap frames of dropping
    below min_persons inside the run (to absorb flicker).
    """
    if not counts:
        return []
    frames = sorted(counts.keys())
    first_f, last_f = min(frames), max(frames)

    # Build a binary signal of "queue active" per frame, with gap-bridging.
    active: list[int] = []
    below_run = 0
    for f in range(first_f, last_f + 1):
        c = counts.get(f, 0)
        if c >= min_persons:
            active.append(1)
            below_run = 0
        else:
            below_run += 1
            # Within max_gap of an active stretch -> still consider active.
            if active and active[-1] == 1 and below_run <= max_gap:
                active.append(1)
            else:
                active.append(0)

    # Walk active signal, find runs >= min_duration.
    events: list[tuple[int, str]] = []
    run_start = None
    for i, a in enumerate(active):
        f_actual = first_f + i
        if a and run_start is None:
            run_start = f_actual
        elif not a and run_start is not None:
            run_len = f_actual - run_start
            if run_len >= min_duration:
                if emit == "alerts":
                    events.append((run_start, "alert"))
                else:
                    events.append((run_start, "queue_start"))
                    events.append((f_actual, "queue_end"))
            run_start = None
    if run_start is not None:
        run_len = (last_f + 1) - run_start
        if run_len >= min_duration:
            if emit == "alerts":
                events.append((run_start, "alert"))
            else:
                events.append((run_start, "queue_start"))
                events.append((last_f, "queue_end"))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="path to seq*.txt OR directory containing seq*/ folders")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-persons", type=int, default=1,
                    help="threshold of simultaneous queueing persons to call it a queue")
    ap.add_argument("--emit", choices=["alerts", "boundaries"], default="alerts",
                    help="alerts: 1 event per queue formation (start). "
                         "boundaries: queue_start + queue_end per formation.")
    ap.add_argument("--min-duration", type=int, default=10,
                    help="queue must be active for >= this many frames (default 10 ~= 0.7s @ 14fps)")
    ap.add_argument("--max-gap", type=int, default=5,
                    help="bridge brief drops below threshold up to N frames (flicker filter)")
    ap.add_argument("--class-id", type=int, default=3,
                    help="CAD class id. Default 3 = Queueing in original Choi 2009 mapping.")
    args = ap.parse_args()

    if args.src.is_file():
        ann_files = [args.src]
    else:
        ann_files = sorted(args.src.rglob("annotations.txt")) or sorted(args.src.rglob("seq*.txt"))

    if not ann_files:
        raise SystemExit(f"no annotation files found under {args.src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[tuple[str, int, str]] = []
    for txt in ann_files:
        seq_id = txt.parent.name if txt.parent.name.startswith("seq") else txt.stem
        counts = parse_seq_annotation(txt, args.class_id)
        evs = emit_events(counts, args.min_persons, args.emit, args.min_duration, args.max_gap)
        for f, ev in evs:
            all_rows.append((seq_id, f, ev))
        print(f"  {seq_id}: {len(evs)} events ({len(counts)} frames with queueing)")

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame", "event"])
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {args.out}: {len(all_rows)} total events across {len(ann_files)} sequences")


if __name__ == "__main__":
    main()

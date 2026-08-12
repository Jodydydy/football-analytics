"""Manual ground-truth labeller for line-crossing events.

Usage:
    python gt_label.py --source test_video.mp4 --line 0.55 --out events_gt.csv

Keys (focus must be on the playback window, not the terminal):
    SPACE  toggle play / pause
    a      step back 1 frame
    d      step forward 1 frame
    A      step back 10 frames (Shift+a)
    D      step forward 10 frames (Shift+d)
    i      mark an IN event at the current frame  (top -> bottom)
    o      mark an OUT event at the current frame (bottom -> top)
    u      undo last event
    s      save without quitting
    q      save and quit

The convention IN = top->bottom matches main.py.
If the file at --out already exists, its events are loaded so labelling can
be resumed across sessions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def draw_hud(frame, frame_idx: int, total: int, paused: bool,
             in_count: int, out_count: int, last_event: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    state = "PAUSED" if paused else "PLAY"
    cv2.putText(frame,
                f"{state}  frame {frame_idx}/{total - 1}  IN {in_count}  OUT {out_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame,
                "SPACE play  a/d step  A/D x10  i=IN  o=OUT  u=undo  s=save  q=save+quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if last_event:
        cv2.putText(frame, last_event, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--line", type=float, default=0.55,
                    help="y-fraction of the visual reference line")
    ap.add_argument("--out", default="events_gt.csv")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ly = int(h * args.line)

    out_path = Path(args.out)
    events: list[tuple[int, str]] = []
    if out_path.exists():
        with out_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                events.append((int(row["frame"]), row["event"]))
        print(f"loaded {len(events)} existing events from {out_path}")

    def save():
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "event"])
            for fr, ev in sorted(events):
                writer.writerow([fr, ev])
        in_n = sum(1 for _, ev in events if ev == "in")
        out_n = sum(1 for _, ev in events if ev == "out")
        print(f"saved {len(events)} events ({in_n} in, {out_n} out) to {out_path}")

    paused = True
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame 0 from {args.source}")
    last_event_msg = ""

    while True:
        in_count = sum(1 for _, ev in events if ev == "in")
        out_count = sum(1 for _, ev in events if ev == "out")
        disp = frame.copy()
        cv2.line(disp, (0, ly), (w, ly), (0, 0, 255), 2)
        disp = draw_hud(disp, frame_idx, total, paused, in_count, out_count, last_event_msg)
        cv2.imshow("gt_label", disp)
        last_event_msg = ""

        wait = 0 if paused else 33  # ~30 FPS during playback
        key = cv2.waitKey(wait) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            paused = True
            new_idx = max(0, frame_idx - 1)
            if new_idx != frame_idx:
                frame_idx = new_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
        elif key == ord('d'):
            paused = True
            if frame_idx < total - 1:
                ok, frame = cap.read()
                if ok:
                    frame_idx += 1
        elif key == ord('A'):
            paused = True
            frame_idx = max(0, frame_idx - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        elif key == ord('D'):
            paused = True
            frame_idx = min(total - 1, frame_idx + 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        elif key == ord('i'):
            events.append((frame_idx, "in"))
            last_event_msg = f"+ IN at frame {frame_idx}"
        elif key == ord('o'):
            events.append((frame_idx, "out"))
            last_event_msg = f"+ OUT at frame {frame_idx}"
        elif key == ord('u'):
            if events:
                fr, ev = events.pop()
                last_event_msg = f"- undo {ev} at frame {fr}"
        elif key == ord('s'):
            save()
            last_event_msg = "saved"
        elif key == ord('q'):
            save()
            break
        elif not paused:
            # auto-advance one frame
            if frame_idx < total - 1:
                ok, frame = cap.read()
                if ok:
                    frame_idx += 1
                else:
                    paused = True
            else:
                paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

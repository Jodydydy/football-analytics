"""
gt_marker.py — ручная разметка эталона по видео, поверх зон, с горячими клавишами.

Накладывает полигоны зон на кадр и пишет события (passer_by / entered / exited /
lingered) в CSV по нажатию клавиши. Этим инструментом размечен весь эталон,
по которому считаются F1 в docs/METRICS.md.

Почему разметка ручная и что это значит для метрик: эталон размечен одним
человеком, поэтому оценки межаннотаторного согласия нет. На спорных событиях
(человек постоял в дверях и ушёл, группа зашла вместе) это реальный источник
смещения — и его нужно называть самому, а не ждать вопроса.

Запуск:
    python eval/gt_marker.py --video video.mp4 --zones zones.json --out gt.csv
    python eval/gt_marker.py ... --start 60     # начать с 60-й секунды
    python eval/gt_marker.py ... --speed 0.5    # замедленное воспроизведение

Hotkeys (показываются в HUD):
    Space     — play / pause
    a / d     — -1 сек / +1 сек
    A / D     — -10 сек / +10 сек  (Shift+a / Shift+d)
    , / .     — -1 кадр / +1 кадр  (как в mpv)
    j / k     — -60 сек / +60 сек
    1         — passer_by L2R       (слева направо)
    2         — passer_by R2L       (справа налево)
    3         — entered             (зашёл в дверь)
    4         — exited              (вышел из двери)
    5         — lingered L2R        (остановился у фасада, шёл слева)
    6         — lingered R2L        (остановился у фасада, шёл справа)
    7         — same as 1, conf=0.7 (uncertain L2R)
    8         — same as 2, conf=0.7 (uncertain R2L)
    u         — undo последний event
    z         — save now (manual)
    h         — toggle overlay (zones on/off)
    + / -     — playback speed up / down
    g         — goto timestamp (ввод в консоли)
    q / ESC   — quit (auto-save)

Auto-save каждые 30 сек wall-clock + при quit. Backup перед каждым start.
CSV формат: timestamp_sec,event_type,direction,confidence,note
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Ensure UTF-8 stdout so cyrillic/emoji prints don't crash on cp1251 consoles
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Данных в репозитории нет — пути задаются аргументами. Значения ниже лишь
# задают удобные умолчания относительно корня проекта.
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FOOTFALL_DATA", ROOT / "data"))
VIDEO = DATA / "video.mp4"
ZONES = DATA / "zones.json"
OUT_CSV = DATA / "gt.csv"
BACKUP_DIR = DATA / "_gt_backups"

ZONE_COLORS = {
    "entrance":       (0, 255, 255),   # жёлтый — дверь
    "facade":         (0, 200, 255),   # оранжевый — лента фасада
    "sidewalk_left":  (255, 120, 80),  # синий — левый тротуар
    "sidewalk_right": (80, 120, 255),  # красный — правый тротуар
}

# hotkey -> (event_type, direction, confidence)
LABELS = {
    ord("1"): ("passer_by", "L2R", 1.0),
    ord("2"): ("passer_by", "R2L", 1.0),
    ord("3"): ("entered",   "",    1.0),
    ord("4"): ("exited",    "",    1.0),
    ord("5"): ("lingered",  "L2R", 1.0),
    ord("6"): ("lingered",  "R2L", 1.0),
    ord("7"): ("passer_by", "L2R", 0.7),
    ord("8"): ("passer_by", "R2L", 0.7),
}

CSV_HEADER = ["timestamp_sec", "event_type", "direction", "confidence", "note"]


def load_zones(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polys = {}
    for name in ZONE_COLORS:
        if name in data and "polygon" in data[name]:
            polys[name] = np.array(data[name]["polygon"], dtype=np.int32)
    return polys


def load_existing(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = r.get("timestamp_sec", "").strip()
            if not ts or ts.startswith("#"):
                continue
            try:
                r["timestamp_sec"] = float(ts)
            except ValueError:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["timestamp_sec"])
    return rows


def save_csv(path: Path, events: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for e in sorted(events, key=lambda r: float(r["timestamp_sec"])):
            w.writerow({k: e.get(k, "") for k in CSV_HEADER})


def backup(path: Path):
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"gt_dvr_{ts}.csv"
    shutil.copy2(path, dst)


def fmt_ts(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def draw_overlay(frame, polys, show_overlay):
    if not show_overlay:
        return frame
    overlay = frame.copy()
    for name, pts in polys.items():
        color = ZONE_COLORS[name]
        cv2.fillPoly(overlay, [pts], color)
    out = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    for name, pts in polys.items():
        color = ZONE_COLORS[name]
        cv2.polylines(out, [pts], True, color, 2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(out, name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def draw_hud(frame, *, ts, total, frame_idx, total_frames, paused,
             speed, count, last_msg, last_msg_until):
    h, w = frame.shape[:2]
    # bottom panel
    panel_h = 78
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, h - panel_h), (w, h - panel_h + 1), (80, 80, 80), -1)

    # progress bar
    bar_y = h - panel_h + 8
    bar_h = 6
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (60, 60, 60), -1)
    pos = int((w - 20) * frame_idx / max(total_frames, 1))
    cv2.rectangle(frame, (10, bar_y), (10 + pos, bar_y + bar_h), (0, 220, 100), -1)

    # left: timestamp
    line1 = f"{fmt_ts(ts)} / {fmt_ts(total)}   frame {frame_idx}/{total_frames}"
    cv2.putText(frame, line1, (10, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # state
    state = "PAUSED" if paused else f"PLAY x{speed:.2f}"
    color = (0, 220, 255) if paused else (0, 220, 100)
    cv2.putText(frame, state, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # right: counters
    counts_text = f"events: {count}"
    (tw, _), _ = cv2.getTextSize(counts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, counts_text, (w - tw - 10, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # mini hotkey legend (top-right)
    legend = [
        "1/2 passer L2R/R2L  3 entered  4 exited",
        "5/6 lingered L2R/R2L  7/8 unsure",
        "Space play  a/d -+1s  A/D -+10s  ,/. frame",
        "u undo  z save  h overlay  q quit",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (w - 460, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # flash message
    if last_msg and time.time() < last_msg_until:
        (tw, th), _ = cv2.getTextSize(last_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (w - tw) // 2
        y = 60
        cv2.rectangle(frame, (x - 12, y - th - 8), (x + tw + 12, y + 10), (0, 100, 0), -1)
        cv2.putText(frame, last_msg, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def goto_dialog(cap, fps, total_frames):
    """Read 'mm:ss' or seconds from console, return frame index."""
    print(">>> goto: enter timestamp as 'mm:ss' or 'sec' (empty = cancel):")
    try:
        s = input("    > ").strip()
    except EOFError:
        return None
    if not s:
        return None
    try:
        if ":" in s:
            mm, ss = s.split(":")
            sec = int(mm) * 60 + float(ss)
        else:
            sec = float(s)
    except ValueError:
        print(f"    [bad input: {s}]")
        return None
    fr = int(sec * fps)
    fr = max(0, min(fr, total_frames - 1))
    return fr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--zones", default=str(ZONES))
    ap.add_argument("--out",   default=str(OUT_CSV))
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start at this timestamp (sec). Default: resume from last event")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Initial playback speed multiplier")
    args = ap.parse_args()

    video_path = Path(args.video)
    zones_path = Path(args.zones)
    out_path = Path(args.out)

    if not video_path.exists():
        sys.exit(f"video not found: {video_path}")
    if not zones_path.exists():
        sys.exit(f"zones not found: {zones_path}")

    polys = load_zones(zones_path)
    print(f"[zones] loaded: {list(polys.keys())}")

    # Backup before start
    backup(out_path)
    events = load_existing(out_path)
    print(f"[csv] existing events: {len(events)} in {out_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"[video] {fps:.2f} fps, {total_frames} frames, {fmt_ts(duration)} total")

    # Decide start position
    if args.start > 0:
        start_frame = int(args.start * fps)
    elif events:
        # resume just after last event
        last_ts = max(float(e["timestamp_sec"]) for e in events)
        start_frame = int(last_ts * fps) + 1
        print(f"[resume] last event @ {fmt_ts(last_ts)} → starting at frame {start_frame}")
    else:
        start_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    # State
    paused = True
    show_overlay = True
    speed = max(0.1, args.speed)
    last_msg = ""
    last_msg_until = 0.0
    last_save_t = time.time()
    auto_save_every = 30  # sec

    win = "GT marker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # We control playback ourselves via wait time per frame
    base_delay_ms = max(1, int(1000 / fps))

    while True:
        # Read current frame (always seek for precision)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            print(f"[video] EOF or read error at frame {current_frame}")
            break

        ts = current_frame / fps
        view = draw_overlay(frame, polys, show_overlay)
        view = draw_hud(view,
                        ts=ts, total=duration,
                        frame_idx=current_frame, total_frames=total_frames,
                        paused=paused, speed=speed,
                        count=len(events),
                        last_msg=last_msg, last_msg_until=last_msg_until)
        cv2.imshow(win, view)

        wait_ms = 30 if paused else max(1, int(base_delay_ms / speed))
        key = cv2.waitKey(wait_ms) & 0xFF

        # Auto-save tick
        if time.time() - last_save_t > auto_save_every:
            save_csv(out_path, events)
            last_save_t = time.time()
            last_msg = f"auto-saved {len(events)} events"
            last_msg_until = time.time() + 1.5

        # No key & not paused → advance
        if key == 255:
            if not paused:
                current_frame = min(current_frame + 1, total_frames - 1)
                if current_frame >= total_frames - 1:
                    paused = True
            continue

        # === Navigation ===
        if key == 32:  # Space
            paused = not paused
        elif key == ord("a"):
            current_frame = max(0, current_frame - int(fps))
            paused = True
        elif key == ord("d"):
            current_frame = min(total_frames - 1, current_frame + int(fps))
            paused = True
        elif key == ord("A"):
            current_frame = max(0, current_frame - int(fps * 10))
            paused = True
        elif key == ord("D"):
            current_frame = min(total_frames - 1, current_frame + int(fps * 10))
            paused = True
        elif key == ord("j"):
            current_frame = max(0, current_frame - int(fps * 60))
            paused = True
        elif key == ord("k"):
            current_frame = min(total_frames - 1, current_frame + int(fps * 60))
            paused = True
        elif key == ord(","):
            current_frame = max(0, current_frame - 1)
            paused = True
        elif key == ord("."):
            current_frame = min(total_frames - 1, current_frame + 1)
            paused = True
        elif key in (ord("+"), ord("=")):
            speed = min(8.0, speed * 1.5)
            last_msg = f"speed x{speed:.2f}"
            last_msg_until = time.time() + 0.8
        elif key == ord("-"):
            speed = max(0.1, speed / 1.5)
            last_msg = f"speed x{speed:.2f}"
            last_msg_until = time.time() + 0.8
        elif key == ord("h"):
            show_overlay = not show_overlay
        elif key == ord("g"):
            target = goto_dialog(cap, fps, total_frames)
            if target is not None:
                current_frame = target
                paused = True

        # === Labels ===
        elif key in LABELS:
            etype, direction, conf = LABELS[key]
            row = {
                "timestamp_sec": round(ts, 2),
                "event_type":    etype,
                "direction":     direction,
                "confidence":    conf,
                "note":          "",
            }
            events.append(row)
            tag = f"{etype}" + (f" {direction}" if direction else "")
            last_msg = f"+ {tag} @ {fmt_ts(ts)} (conf {conf})"
            last_msg_until = time.time() + 1.2
            print(f"[+] {fmt_ts(ts)}  {etype:10s} {direction:3s} conf={conf}")

        # === Edit ===
        elif key == ord("u"):
            if events:
                gone = events.pop()
                last_msg = f"undo: {gone['event_type']} @ {fmt_ts(float(gone['timestamp_sec']))}"
                last_msg_until = time.time() + 1.5
                print(f"[-] undo: {gone}")
        elif key == ord("z"):
            save_csv(out_path, events)
            last_save_t = time.time()
            last_msg = f"saved {len(events)} events"
            last_msg_until = time.time() + 1.5
            print(f"[save] {len(events)} events -> {out_path}")

        # === Quit ===
        elif key in (ord("q"), 27):  # q or ESC
            break

    save_csv(out_path, events)
    print(f"\n[final] saved {len(events)} events -> {out_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

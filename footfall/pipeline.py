"""
Сборка пайплайна: кадр -> детекция -> трекинг -> зачёт -> лог событий.

Порядок стадий не произволен и не переставляется:

    источник     кадр за кадром
    детекция     что на кадре сейчас, без памяти о прошлом
    трекинг      связывание с прошлым, появляется track_id
    зачёт        событие, требующее знания траектории
    лог          append-only CSV, единственный источник правды на точке
    постобработка чистка ложных событий по готовому логу

Счётчик не может работать без track_id, поэтому трекинг обязан идти после
детекции и до зачёта. Постобработке нужен весь лог целиком, поэтому она
идёт после цикла.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from .counters import (SingleLineCounter, TripwireCounter,
                       ZoneAbsenceCounter, ZoneDwellCounter)
from .detect import Detector
from .ensemble import merge_person_and_head
from .postprocess import apply_cooldown, cancel_roundtrips, count_directions
from .sources import open_source
from .track import Tracker
from .zones import load_zone, parse_xy

ANCHORS = {
    "bottom-center": sv.Position.BOTTOM_CENTER,
    "center": sv.Position.CENTER,
    "top-center": sv.Position.TOP_CENTER,
}


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Выровнять освещённость перед детекцией.

    CLAHE — адаптивное выравнивание гистограммы по каналу яркости в LAB.
    Нужно на входных дверях: улица за спиной человека даёт контровой свет,
    силуэт уходит в тень, детектор его теряет. Обычное выравнивание
    гистограммы по всему кадру здесь не помогает — пересвет у двери
    перетягивает всю шкалу. CLAHE работает по плиткам, поэтому вытягивает
    тёмную область, не убивая светлую.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def build_counter(args, w: int, h: int, fps: float) -> tuple[Any, str]:
    """Выбрать и собрать счётчик по аргументам. Вернуть (счётчик, описание)."""
    anchor = ANCHORS[args.anchor]

    if args.zone:
        polygon = load_zone(args.zone)
        dwell_frames = max(1, round(args.dwell * fps))
        queue_alert_frames = (max(1, round(args.queue_duration * fps))
                              if args.queue_threshold > 0 else 0)
        session_min = (max(1, round(args.session_min_seconds * fps))
                       if args.count_mode == "session-max" else 0)
        counter = ZoneDwellCounter(polygon, anchor, dwell_frames,
                                   queue_threshold=args.queue_threshold,
                                   queue_alert_frames=queue_alert_frames,
                                   count_mode=args.count_mode,
                                   session_min_frames=session_min)
        msg = (f"zone-dwell polygon={Path(args.zone).name} "
               f"dwell={args.dwell}s ({dwell_frames}f @ {fps:.1f}fps)")
        if args.queue_threshold > 0:
            msg += (f"  queue-alert: >={args.queue_threshold} for "
                    f"{args.queue_duration}s ({queue_alert_frames}f)")
        return counter, msg

    if (args.start is None) != (args.end is None):
        raise SystemExit("--start and --end must be given together")

    if args.start is not None:
        sx, sy = parse_xy(args.start)
        ex, ey = parse_xy(args.end)
        start_pt, end_pt = sv.Point(sx, sy), sv.Point(ex, ey)
        line_desc = f"({sx},{sy})->({ex},{ey})"
    else:
        ly = int(h * args.line)
        start_pt, end_pt = sv.Point(0, ly), sv.Point(w, ly)
        line_desc = f"horizontal y={ly}"

    if args.gap > 0:
        delta = max(int(h * args.gap / 2), 1)
        counter = TripwireCounter(start_pt, end_pt, delta, anchor,
                                  args.min_cross, args.max_gap_frames)
        msg = (f"tripwire pair {line_desc} delta={delta}px "
               f"max_gap={args.max_gap_frames}f")
    else:
        counter = SingleLineCounter(start_pt, end_pt, anchor, args.min_cross,
                                    dedup=not args.no_dedup)
        msg = f"single line {line_desc}{' (no-dedup)' if args.no_dedup else ''}"
    return counter, msg


def run(args) -> None:
    detector = Detector(args.model, device=args.device or None,
                        conf=args.conf, imgsz=args.imgsz, augment=args.tta)
    head_detector = (Detector(args.ensemble_head, device=args.device or None,
                              conf=args.conf, imgsz=args.imgsz, augment=args.tta)
                     if args.ensemble_head else None)
    tracker = Tracker(lost_track_buffer=args.track_buffer,
                      minimum_consecutive_frames=args.track_min_frames)

    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    counter, mode_msg = build_counter(args, w, h, fps)

    absence_counter = None
    if args.absence_zone:
        absence_frames = max(1, round(args.absence_duration * fps))
        absence_counter = ZoneAbsenceCounter(
            load_zone(args.absence_zone), ANCHORS[args.anchor], absence_frames,
            fire_from_start=args.absence_fire_from_start)
        mode_msg += (f"  +absence-zone={Path(args.absence_zone).name} "
                     f"threshold={args.absence_duration}s ({absence_frames}f)")

    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()
    trace_annot = sv.TraceAnnotator()
    line_annot = sv.LineZoneAnnotator(thickness=3, text_thickness=2,
                                      text_scale=0.8,
                                      display_in_count=False,
                                      display_out_count=False)

    log_path = Path(args.log)
    new_file = not log_path.exists()
    # запоминаем длину лога до старта: постобработка должна трогать только
    # события ЭТОЙ сессии, а не предыдущих прогонов в тот же файл
    pre_session_lines = (sum(1 for _ in log_path.open("r", newline=""))
                         if log_path.exists() else 0)
    log_file = log_path.open("a", newline="")
    writer = csv.writer(log_file)
    if new_file:
        writer.writerow(["frame", "timestamp", "event", "track_id"])
        pre_session_lines = 1

    out_writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
        if not out_writer.isOpened():
            raise SystemExit(f"cannot open {args.out_video} for writing")

    positions_writer = positions_file = None
    positions_every = max(1, args.positions_every)
    if args.positions_log:
        pos_path = Path(args.positions_log)
        pos_new = not pos_path.exists()
        positions_file = pos_path.open("a", newline="")
        positions_writer = csv.writer(positions_file)
        if pos_new:
            positions_writer.writerow(["frame", "timestamp", "track_id", "x", "y"])

    max_frames = round(args.max_seconds * fps) if args.max_seconds > 0 else 0
    warmup_frames = round(args.warmup_seconds * fps)

    print(f"source={args.source}  frame={w}x{h}  {mode_msg}  log={log_path}")
    if out_writer is not None:
        print(f"writing annotated video -> {args.out_video}")
    if max_frames:
        print(f"limit: first {args.max_seconds}s ({max_frames} frames @ {fps:.1f}fps)")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            yolo_input = apply_clahe(frame) if args.preproc == "clahe" else frame
            detections = detector.detect(yolo_input)
            if head_detector is not None:
                detections = merge_person_and_head(
                    detections, head_detector.detect(yolo_input),
                    iou_drop=args.ensemble_iou)
            detections = tracker.update(detections)

            in_tids, out_tids = counter.update(detections)
            if args.invert_direction:
                in_tids, out_tids = out_tids, in_tids

            queue_alert = (counter.pop_queue_alert()
                           if isinstance(counter, ZoneDwellCounter) else None)
            absence_alert = (absence_counter.update(detections)
                             if absence_counter is not None else None)

            # разогрев: в первых кадрах детектор даёт всплеск ложных срабатываний,
            # а ByteTrack ещё не стабилизировал треки — глушим все события
            if frame_idx <= warmup_frames:
                in_tids, out_tids = [], []
                queue_alert = absence_alert = None

            if (positions_writer is not None
                    and frame_idx % positions_every == 0
                    and detections.tracker_id is not None
                    and len(detections) > 0):
                now = datetime.now().isoformat(timespec="seconds")
                xy = detections.xyxy
                cxs = (xy[:, 0] + xy[:, 2]) / 2
                cys = (xy[:, 1] + xy[:, 3]) / 2
                for i, raw_tid in enumerate(detections.tracker_id):
                    positions_writer.writerow([frame_idx, now, int(raw_tid),
                                               int(cxs[i]), int(cys[i])])
                positions_file.flush()

            if in_tids or out_tids or queue_alert is not None or absence_alert is not None:
                now = datetime.now().isoformat(timespec="seconds")
                for tid in in_tids:
                    writer.writerow([frame_idx, now, "in", tid])
                for tid in out_tids:
                    writer.writerow([frame_idx, now, "out", tid])
                if queue_alert is not None:
                    writer.writerow([frame_idx, now, "queue_alert", queue_alert])
                    print(f"  [QUEUE ALERT] frame {frame_idx}: {queue_alert} in zone")
                if absence_alert is not None:
                    secs = round(absence_alert / fps)
                    writer.writerow([frame_idx, now, "absence_alert", secs])
                    print(f"  [ABSENCE ALERT] frame {frame_idx}: empty {secs}s")
                log_file.flush()

            if args.show or out_writer is not None:
                frame = _annotate(frame, detections, counter, absence_counter,
                                  trace_annot, box_annot, label_annot,
                                  line_annot, fps)
                if out_writer is not None:
                    out_writer.write(frame)
                if args.show:
                    cv2.imshow("footfall", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        log_file.close()
        if positions_file is not None:
            positions_file.close()
        if out_writer is not None:
            out_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    _postprocess_and_report(args, counter, log_path, pre_session_lines)


def _annotate(frame, detections, counter, absence_counter,
              trace_annot, box_annot, label_annot, line_annot, fps):
    tids = detections.tracker_id if detections.tracker_id is not None else []
    labels = [f"#{int(t)}" for t in tids]
    frame = trace_annot.annotate(frame, detections)
    frame = box_annot.annotate(frame, detections)
    if labels:
        frame = label_annot.annotate(frame, detections, labels)

    if isinstance(counter, ZoneDwellCounter):
        frame = counter.annotate(frame, detections, fps)
        hud = f"CUSTOMERS {counter.in_count}  NOW {counter.current_occupants}"
        if counter.queue_threshold > 0:
            hud += f"  (alert>={counter.queue_threshold})"
    else:
        frame = counter.annotate(frame, line_annot)
        hud = (f"IN {counter.in_count}  OUT {counter.out_count}  "
               f"NET {counter.in_count - counter.out_count}")

    if absence_counter is not None:
        frame = absence_counter.annotate(frame, fps)
    cv2.putText(frame, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (49, 33, 239), 2)
    return frame


def _postprocess_and_report(args, counter, log_path: Path,
                            pre_session_lines: int) -> None:
    """Почистить лог и напечатать итог. Зонному счётчику чистка не нужна."""
    is_zone = isinstance(counter, ZoneDwellCounter)

    if args.cancel_window > 0 and not is_zone:
        n = cancel_roundtrips(log_path, pre_session_lines, args.cancel_window)
        if n:
            print(f"cancelled {n} round-trip event(s) "
                  f"(window={args.cancel_window} frames)")

    if args.per_track_cooldown > 0 and not is_zone:
        n = apply_cooldown(log_path, pre_session_lines, args.per_track_cooldown)
        if n:
            print(f"cooldown dropped {n} same-direction repeat(s) "
                  f"(window={args.per_track_cooldown} frames)")

    if is_zone:
        print(f"final: CUSTOMERS={counter.in_count}")
    else:
        in_n, out_n = count_directions(log_path, pre_session_lines)
        print(f"final: IN={in_n}  OUT={out_n}  NET={in_n - out_n}")

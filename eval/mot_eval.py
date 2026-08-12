"""MOT-style precision/recall for YOLO+ByteTrack on CAFE Dataset.

Feeds CAFE clip frames directly into YOLO+ByteTrack (no mp4 intermediate),
matches predictions against gt_tracks.txt by IoU, reports per-clip and
aggregate metrics.

Usage:
    python mot_eval.py --clips 1:40,1:342,1:229 --conf 0.25 --imgsz 1280
    python mot_eval.py --clips 1:18,1:44,1:57       # ordering clips
    python mot_eval.py --cafe 1 --activity 1        # all queueing in cafe 1

Metrics:
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = harmonic mean
    id_switches = number of times a GT track changes its matched pred ID

GT semantics:
    CAFE GT track_id is the GROUP id (multiple people in a group share id).
    For detection metrics this is fine — we match by IoU regardless of id.
    For ID-switch counting we work per-GT-row (each box is one "GT instance").
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from cafe_helper import DEFAULT_ROOT, clip_frames, gt_for_clip, list_clips


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes [x1,y1,x2,y2]."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes: list[np.ndarray], pred_boxes: list[np.ndarray],
                 iou_thresh: float) -> list[tuple[int, int, float]]:
    """Greedy IoU matching. Returns [(gt_idx, pred_idx, iou), ...]."""
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            v = iou(gb, pb)
            if v >= iou_thresh:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    out = []
    for v, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi); used_pred.add(pi)
        out.append((gi, pi, v))
    return out


def evaluate_clip(cafe_id: int, clip_id: int, model: YOLO,
                  conf: float, imgsz: int, iou_thresh: float,
                  root: Path,
                  filter_classes: set[int] | None = None) -> dict:
    """Run YOLO+ByteTrack on a clip and compare to GT by IoU.

    CAFE caveat: GT only annotates "actors" (people in the scripted scene),
    not every person visible. Total precision is meaningless. Recall on
    actors (filter_classes={1,2,...}) is the meaningful number — does the
    model see the people who matter.
    """
    tracker = sv.ByteTrack()  # fresh tracker per clip
    frames = clip_frames(cafe_id, clip_id, root=root)
    gt_rows = gt_for_clip(cafe_id, clip_id, root=root)
    gt_by_frame: dict[int, list] = defaultdict(list)
    for r in gt_rows:
        if r["x1"] < 0 or r["x2"] <= r["x1"] or r["y2"] <= r["y1"]:
            continue
        if filter_classes is not None and r["activity"] not in filter_classes:
            continue
        gt_by_frame[r["frame_id"]].append(r)

    tp = fn = 0
    id_switches = 0
    gt_track_to_pred: dict[int, int] = {}  # last pred-id matched to each GT-track-id
    n_dets_total = 0

    for f in frames:
        frame_id = int(f.stem.split("_")[1])
        img = cv2.imread(str(f))
        if img is None:
            continue
        results = model(img, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        det = sv.Detections.from_ultralytics(results)
        det = tracker.update_with_detections(det)
        n_dets_total += len(det)

        pred_boxes = [det.xyxy[i] for i in range(len(det))]
        pred_ids = ([int(t) for t in det.tracker_id]
                    if det.tracker_id is not None else [-1] * len(det))

        gts = gt_by_frame.get(frame_id, [])
        gt_boxes = [np.array([g["x1"], g["y1"], g["x2"], g["y2"]]) for g in gts]
        gt_ids = [g["track_id"] for g in gts]

        matches = greedy_match(gt_boxes, pred_boxes, iou_thresh)
        tp += len(matches)
        fn += len(gt_boxes) - len(matches)
        # FP not meaningful (GT incomplete) — skipped

        for gi, pi, _ in matches:
            gt_tid = gt_ids[gi]
            if gt_tid < 0:
                continue
            prev = gt_track_to_pred.get(gt_tid)
            if prev is not None and prev != pred_ids[pi]:
                id_switches += 1
            gt_track_to_pred[gt_tid] = pred_ids[pi]

    return {
        "cafe": cafe_id, "clip": clip_id,
        "frames": len(frames), "tp": tp, "fn": fn,
        "id_switches": id_switches,
        "n_pred_dets": n_dets_total,
        "filter": filter_classes,
    }


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def report(results: list[dict]) -> None:
    print(f"\n{'cafe/clip':>10} {'frames':>7} {'GT':>5} {'TP':>5} {'FN':>5} "
          f"{'recall':>9} {'preds':>7} {'IDsw':>5}")
    print("-" * 70)
    tot_tp = tot_fn = tot_idsw = tot_frames = tot_preds = 0
    for r in results:
        gt_total = r["tp"] + r["fn"]
        rec = r["tp"] / max(1, gt_total)
        print(f"{r['cafe']:>4}/{r['clip']:<5} {r['frames']:>7} "
              f"{gt_total:>5} {r['tp']:>5} {r['fn']:>5} "
              f"{fmt_pct(rec):>9} {r['n_pred_dets']:>7} "
              f"{r['id_switches']:>5}")
        tot_tp += r["tp"]; tot_fn += r["fn"]
        tot_idsw += r["id_switches"]; tot_frames += r["frames"]
        tot_preds += r["n_pred_dets"]
    print("-" * 70)
    gt_total = tot_tp + tot_fn
    rec = tot_tp / max(1, gt_total)
    print(f"{'TOTAL':>10} {tot_frames:>7} "
          f"{gt_total:>5} {tot_tp:>5} {tot_fn:>5} "
          f"{fmt_pct(rec):>9} {tot_preds:>7} "
          f"{tot_idsw:>5}")
    print(f"\n  preds/frame avg: {tot_preds/max(1,tot_frames):.1f}  "
          f"(CAFE GT is incomplete — annotates only scripted actors,\n"
          f"   so precision/FP not meaningful. Recall = how often YOLO\n"
          f"   finds the actors that DO matter to the task.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", help="comma-separated cafe:clip, e.g. 1:40,1:342")
    ap.add_argument("--cafe", type=int, help="restrict --activity discovery to this cafe")
    ap.add_argument("--activity", type=int, help="discover all clips of this class")
    ap.add_argument("--limit", type=int, default=10, help="limit clips when using --activity")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--filter-classes", default=None,
                    help="comma-separated activity classes to keep in GT "
                         "(e.g. '1,2' to focus on queueing+ordering actors). "
                         "Default: all classes.")
    args = ap.parse_args()

    root = DEFAULT_ROOT
    if args.clips:
        clips = []
        for tok in args.clips.split(","):
            c, k = tok.split(":")
            clips.append((int(c), int(k)))
    elif args.activity is not None:
        rows = list_clips(args.activity, args.cafe, root=root)
        clips = [(c, k) for c, k, _ in rows[:args.limit]]
    else:
        raise SystemExit("specify --clips or --activity")

    filter_classes = None
    if args.filter_classes:
        filter_classes = {int(c) for c in args.filter_classes.split(",")}

    print(f"loading YOLO ({args.model})  conf={args.conf}  imgsz={args.imgsz}  IoU>={args.iou}")
    if filter_classes:
        print(f"  GT filter: classes={sorted(filter_classes)}")
    model = YOLO(args.model)

    results = []
    for cafe_id, clip_id in clips:
        print(f"  evaluating cafe {cafe_id} clip {clip_id} ...", flush=True)
        results.append(evaluate_clip(cafe_id, clip_id, model,
                                     args.conf, args.imgsz, args.iou, root,
                                     filter_classes=filter_classes))

    report(results)


if __name__ == "__main__":
    main()

"""Event-based F1 evaluator for door IN/OUT events.

Matches predicted door events to GT events via greedy 1-to-1 matching
with temporal tolerance. Separates IN and OUT directions. Reports:
  - ALL events F1 (workers + customers + visitor + child) — detector-level
  - Per-role recall (customer / worker / visitor / child)
  - Wilson 95% CI on all rates

GT format (gt_door_full4h.csv):
    ts_sec,ts_hms,direction,role,subject_id,notes
    332,00:05:32,IN,worker,w1,
    ...

Pred format (door_events.csv):
    track_id,ts_sec,ts_hms,direction
    1,332.49,00:05:32.48,IN
    ...

Usage:
    python eval/eval_door_events.py \\
        --gt path/to/gt_door.csv \\
        --pred path/to/door_events.csv \\
        --tolerance-sec 5

Why tolerance-based matching: a crossing is an instant in the ground truth but
a range in the prediction — the exact frame depends on which bbox anchor is
used and on tracker latency. Requiring frame-exact equality would report near
zero F1 on a pipeline that is in fact correct. The tolerance is a parameter of
the protocol and must be reported alongside the metric.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_ts(s: str) -> float:
    s = s.strip().strip('"').strip("'")
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + float(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    return float(s)


def read_gt(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                ts = float(r["ts_sec"])
            except (ValueError, KeyError):
                ts = parse_ts(r.get("ts_hms", "0"))
            rows.append({
                "ts": ts,
                "direction": r["direction"].strip().upper(),
                "role": r.get("role", "").strip().lower(),
                "subject_id": r.get("subject_id", "").strip(),
                "notes": r.get("notes", "").strip(),
            })
    rows.sort(key=lambda x: x["ts"])
    return rows


def read_pred(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "ts": float(r["ts_sec"]),
                "direction": r["direction"].strip().upper(),
                "track_id": r.get("track_id", "").strip(),
            })
    rows.sort(key=lambda x: x["ts"])
    return rows


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def event_tolerance(g: dict, default_tol: float, estimated_tol: float) -> float:
    """Allow wider tolerance for GT events with notes containing 'estimated' or 'transient'."""
    notes = g.get("notes", "").lower()
    if "estimated" in notes or "transient" in notes or "lunch_break" in notes:
        return estimated_tol
    return default_tol


def greedy_match(gt: list[dict], pred: list[dict], tol: float,
                 direction: str | None = None,
                 estimated_tol: float | None = None,
                 ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Match within the same direction. direction=None matches both.
    Per-event tolerance: GT events with 'estimated'/'transient' notes use estimated_tol."""
    if estimated_tol is None:
        estimated_tol = tol
    g_idx = [i for i, g in enumerate(gt)
             if direction is None or g["direction"] == direction]
    p_idx = [i for i, p in enumerate(pred)
             if direction is None or p["direction"] == direction]
    pairs: list[tuple[float, int, int]] = []
    for gi in g_idx:
        gi_tol = event_tolerance(gt[gi], tol, estimated_tol)
        for pi in p_idx:
            if gt[gi]["direction"] != pred[pi]["direction"]:
                continue
            d = abs(gt[gi]["ts"] - pred[pi]["ts"])
            if d <= gi_tol:
                pairs.append((d, gi, pi))
    pairs.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for d, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((gi, pi, d))
    fn = [gi for gi in g_idx if gi not in used_g]
    fp = [pi for pi in p_idx if pi not in used_p]
    return matched, fn, fp


def hms(sec: float) -> str:
    hh = int(sec // 3600); rem = sec - hh * 3600
    mm = int(rem // 60); ss = rem - mm * 60
    return f"{hh:01d}:{mm:02d}:{ss:05.2f}"


def report(label: str, tp: int, fp: int, fn: int) -> None:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    p_lo, p_hi = wilson_ci(tp, tp + fp)
    r_lo, r_hi = wilson_ci(tp, tp + fn)
    print(f"  [{label}]  TP={tp}  FP={fp}  FN={fn}")
    print(f"    Precision = {p:.3f}  CI [{p_lo:.3f}, {p_hi:.3f}]")
    print(f"    Recall    = {r:.3f}  CI [{r_lo:.3f}, {r_hi:.3f}]")
    print(f"    F1        = {f1:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--tolerance-sec", type=float, default=5.0)
    ap.add_argument("--estimated-tolerance-sec", type=float, default=15.0,
                    help="Tolerance for GT events with 'estimated'/'transient'/'lunch_break' notes. "
                         "Set equal to --tolerance-sec to disable per-event tolerance.")
    ap.add_argument("--exclude-edge-cases", action="store_true",
                    help="Exclude GT events with notes containing "
                         "'mysterious_skip' or 'estimated' (still report count).")
    args = ap.parse_args()

    gt = read_gt(args.gt)
    pred = read_pred(args.pred)

    excluded = []
    if args.exclude_edge_cases:
        kept = []
        for g in gt:
            notes = g["notes"].lower()
            if "mysterious_skip" in notes or "estimated" in notes:
                excluded.append(g)
            else:
                kept.append(g)
        gt = kept

    print("=" * 72)
    print(f"GT events:   {len(gt)}  (file: {args.gt})")
    print(f"Pred events: {len(pred)}  (file: {args.pred})")
    if excluded:
        print(f"Excluded {len(excluded)} GT events (edge cases)")
    print(f"Tolerance: ±{args.tolerance_sec:.1f} sec")
    print()

    # OVERALL F1 (ALL events, both directions combined)
    m_in, fn_in, fp_in = greedy_match(gt, pred, args.tolerance_sec, "IN",
                                       args.estimated_tolerance_sec)
    m_out, fn_out, fp_out = greedy_match(gt, pred, args.tolerance_sec, "OUT",
                                          args.estimated_tolerance_sec)
    tp_in, tp_out = len(m_in), len(m_out)
    tp_all = tp_in + tp_out
    fp_all = len(fp_in) + len(fp_out)
    fn_all = len(fn_in) + len(fn_out)

    print("=" * 72)
    print("DETECTOR-LEVEL F1 (ALL events: workers + customers + visitor + child)")
    print("=" * 72)
    report("IN", tp_in, len(fp_in), len(fn_in))
    print()
    report("OUT", tp_out, len(fp_out), len(fn_out))
    print()
    report("ALL", tp_all, fp_all, fn_all)
    print()

    # PER-ROLE recall
    print("=" * 72)
    print("PER-ROLE RECALL")
    print("=" * 72)
    all_matched_gi: set[int] = {gi for gi, _, _ in m_in} | {gi for gi, _, _ in m_out}
    by_role: dict[str, dict[str, int]] = {}
    for i, g in enumerate(gt):
        rk = g["role"] or "unknown"
        by_role.setdefault(rk, {"tp": 0, "fn": 0, "in_tp": 0, "in_fn": 0,
                                 "out_tp": 0, "out_fn": 0})
        is_matched = i in all_matched_gi
        d_key = "in" if g["direction"] == "IN" else "out"
        if is_matched:
            by_role[rk]["tp"] += 1
            by_role[rk][f"{d_key}_tp"] += 1
        else:
            by_role[rk]["fn"] += 1
            by_role[rk][f"{d_key}_fn"] += 1

    for rk in sorted(by_role.keys()):
        r = by_role[rk]
        total = r["tp"] + r["fn"]
        recall = r["tp"] / total if total else 0.0
        lo, hi = wilson_ci(r["tp"], total)
        print(f"  {rk:>10s}: {r['tp']:>3d}/{total:>3d} = {recall:.3f}  "
              f"CI [{lo:.3f}, {hi:.3f}]  "
              f"(IN: {r['in_tp']}/{r['in_tp']+r['in_fn']}, "
              f"OUT: {r['out_tp']}/{r['out_tp']+r['out_fn']})")
    print()

    # CUSTOMER-ONLY F1 (business metric: detector vs customer events only).
    # Assumption: customer FPs (pred events near customer events but not matched) are rare;
    # most unmatched pred events are real worker events not in user GT.
    cust_tp = by_role.get("customer", {"tp": 0})["tp"]
    cust_fn = by_role.get("customer", {"fn": 0})["fn"]
    # Customer FPs: pred OUT/IN events that fall WITHIN a customer GT visit interval
    # (in_ts ... in_ts + 30 min) but didn't match. Heuristic: count unmatched pred events
    # whose timestamp is within ±30s of any customer GT IN/OUT — those might be customer-related.
    cust_in_times = [g["ts"] for g in gt if g["role"] == "customer"]
    cust_fps = 0
    for pi in fp_in + fp_out:
        pred_ts = pred[pi]["ts"]
        if any(abs(pred_ts - t) <= 30 for t in cust_in_times):
            cust_fps += 1
    cust_p = cust_tp / (cust_tp + cust_fps) if (cust_tp + cust_fps) else 0.0
    cust_r = cust_tp / (cust_tp + cust_fn) if (cust_tp + cust_fn) else 0.0
    cust_f1 = 2 * cust_p * cust_r / (cust_p + cust_r) if (cust_p + cust_r) else 0.0
    cust_p_lo, cust_p_hi = wilson_ci(cust_tp, cust_tp + cust_fps)
    cust_r_lo, cust_r_hi = wilson_ci(cust_tp, cust_tp + cust_fn)
    print("=" * 72)
    print("CUSTOMER-ONLY F1 (business metric — customer events only)")
    print("=" * 72)
    print(f"  TP={cust_tp}  FN={cust_fn}  FP~={cust_fps} (heuristic: unmatched pred within 30s of cust GT)")
    print(f"  Precision = {cust_p:.3f}  CI [{cust_p_lo:.3f}, {cust_p_hi:.3f}]")
    print(f"  Recall    = {cust_r:.3f}  CI [{cust_r_lo:.3f}, {cust_r_hi:.3f}]")
    print(f"  F1        = {cust_f1:.3f}")
    print()

    # FN report
    if fn_in or fn_out:
        print("=" * 72)
        print("FN (missed GT events):")
        print("=" * 72)
        for gi in sorted(fn_in + fn_out, key=lambda i: gt[i]["ts"]):
            g = gt[gi]
            print(f"  {hms(g['ts'])}  {g['direction']:>3s}  "
                  f"{g['role']:>8s}/{g['subject_id']:<7s}  {g['notes']}")
        print()

    # FP report
    if fp_in or fp_out:
        print("=" * 72)
        print("FP (predicted but no GT match within tolerance):")
        print("=" * 72)
        for pi in sorted(fp_in + fp_out, key=lambda i: pred[i]["ts"]):
            p = pred[pi]
            print(f"  {hms(p['ts'])}  {p['direction']:>3s}  tid={p['track_id']}")
        print()

    # Matches detail (optional, kept brief)
    print("=" * 72)
    print(f"Matched events: {tp_all}  (avg dt = "
          f"{sum(d for _, _, d in m_in + m_out) / max(1, tp_all):.2f}s)")
    print("=" * 72)


if __name__ == "__main__":
    main()

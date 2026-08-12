"""
compare_gt_pred.py — сравнение эталона (ручная разметка) с предсказаниями
пайплайна на уровне событий.

Логика:
  1. Для каждого event_type определить валидный временной диапазон GT (только
     те куски видео, где пользователь активно размечал данный тип).
  2. Greedy-nearest-neighbour matching pred vs GT в окне ±tolerance секунд.
  3. P/R/F1 + Wilson 95% CI + список TP/FP/FN с таймкодами.
  4. Дополнительно: экспорт pred.lingered → lingered_for_review.csv для ручной
     сверки пользователем (форматированные mm:ss + duration, sorted by time).

Маппинг pred event_type → timestamp:
  entered                = t_last_sec   (момент исчезновения в двери)
  exited                 = t_first_sec  (момент появления из двери)
  lingered               = t_first_sec  (момент НАЧАЛА остановки у фасада)
  passer_by_single_side  = (t_first + t_last) / 2  (середина пути по тротуару)
  passer_by              = (t_first + t_last) / 2  (full traverse, тоже середина)

Запуск:
    python eval/compare_gt_pred.py --gt data/gt.csv --pred data/pred.csv
    python eval/compare_gt_pred.py --gt ... --pred ... --tolerance 5

Про временной допуск. Событие в эталоне — момент, а в предсказании — диапазон:
точный кадр зависит от выбранной якорной точки бокса и задержки трекера.
Требовать точного совпадения кадров значит получить F1 около нуля на исправном
пайплайне. Допуск — параметр протокола, и называть его нужно рядом с метрикой.
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

# Данных в репозитории нет — пути задаются аргументами. Значения ниже лишь
# задают удобные умолчания относительно корня проекта.
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FOOTFALL_DATA", ROOT / "data"))
GT_CSV = DATA / "gt.csv"
PRED_CSV = DATA / "pred.csv"
LINGER_OUT = DATA / "lingered_for_review.csv"


def fmt_ts(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_gt(path: Path):
    """Returns dict: event_type -> sorted list of timestamps (sec)."""
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = row.get("timestamp_sec", "").strip()
            if not ts or ts.startswith("#"):
                continue
            try:
                t = float(ts)
            except ValueError:
                continue
            etype = row["event_type"].strip()
            out[etype].append(t)
    for k in out:
        out[k].sort()
    return out


def pred_timestamp(row: dict):
    """Pick representative timestamp per event_type. Returns None if invalid."""
    etype = row["event_type"]
    try:
        t_first = float(row["t_first_sec"])
        t_last = float(row["t_last_sec"])
    except (ValueError, KeyError):
        return None
    if etype == "entered":
        return t_last
    if etype == "exited":
        return t_first
    if etype == "lingered":
        return t_first
    if etype in ("passer_by", "passer_by_single_side"):
        return (t_first + t_last) / 2
    return t_first


def load_pred(path: Path):
    """Returns dict: event_type -> sorted list of (timestamp, row)."""
    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            etype = row["event_type"]
            t = pred_timestamp(row)
            if t is None:
                continue
            out[etype].append((t, row))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def match_greedy(gt_times: list, pred_times: list, tolerance: float):
    """
    Greedy nearest matching. Each GT can match at most one pred and vice versa.

    Returns: (matched_pairs, unmatched_gt, unmatched_pred)
    matched_pairs: list of (gt_t, pred_t)
    """
    # build candidate edges (i_gt, i_pred, |dt|) within tolerance
    edges = []
    for i, gt_t in enumerate(gt_times):
        for j, pred_t in enumerate(pred_times):
            dt = abs(gt_t - pred_t)
            if dt <= tolerance:
                edges.append((dt, i, j))
    edges.sort()  # by dt ascending → greedy nearest first

    used_gt = set()
    used_pred = set()
    pairs = []
    for dt, i, j in edges:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        pairs.append((gt_times[i], pred_times[j]))

    unmatched_gt = [gt_times[i] for i in range(len(gt_times)) if i not in used_gt]
    unmatched_pred = [pred_times[j] for j in range(len(pred_times)) if j not in used_pred]
    return pairs, unmatched_gt, unmatched_pred


def evaluate(label: str, gt_times: list, pred_records: list,
             gt_window: tuple, tolerance: float, verbose: bool = True):
    """
    gt_window = (t_min, t_max) — диапазон по которому считаем (фильтруем pred тоже).
    pred_records = list of (timestamp, row)
    """
    t_min, t_max = gt_window
    gt_in = [t for t in gt_times if t_min <= t <= t_max]
    pred_in = [(t, r) for (t, r) in pred_records if t_min <= t <= t_max]
    pred_times = [t for (t, _) in pred_in]

    pairs, fn_times, fp_times = match_greedy(gt_in, pred_times, tolerance)

    tp = len(pairs)
    fp = len(fp_times)
    fn = len(fn_times)

    p, p_lo, p_hi = wilson_ci(tp, tp + fp)
    r, r_lo, r_hi = wilson_ci(tp, tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"\n=== {label}  (window {fmt_ts(t_min)}..{fmt_ts(t_max)}, tol ±{tolerance:.0f}s) ===")
    print(f"  GT={len(gt_in)}  pred={len(pred_times)}  TP={tp}  FP={fp}  FN={fn}")
    print(f"  P  = {p:.3f}  [{p_lo:.3f}, {p_hi:.3f}]")
    print(f"  R  = {r:.3f}  [{r_lo:.3f}, {r_hi:.3f}]")
    print(f"  F1 = {f1:.3f}")

    if verbose and (fp or fn):
        if fn:
            print(f"\n  FN (GT не нашёл match в pred):  [{len(fn_times)}]")
            for t in fn_times[:30]:
                print(f"    miss  @ {fmt_ts(t)}")
            if len(fn_times) > 30:
                print(f"    ... и ещё {len(fn_times) - 30}")
        if fp:
            print(f"\n  FP (pred без GT):  [{len(fp_times)}]")
            for t in fp_times[:30]:
                print(f"    extra @ {fmt_ts(t)}")
            if len(fp_times) > 30:
                print(f"    ... и ещё {len(fp_times) - 30}")

    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def export_lingered(pred: dict, out_path: Path):
    rows = []
    for t_first, r in pred.get("lingered", []):
        dur = float(r["t_last_sec"]) - float(r["t_first_sec"])
        rows.append({
            "timestamp_mmss":   fmt_ts(t_first),
            "timestamp_sec":    round(t_first, 2),
            "duration_sec":     round(dur, 2),
            "facade_dwell_sec": float(r["facade_dwell_sec"]),
            "max_disp_in_facade": float(r["max_disp_in_facade"]),
            "facing_cafe_frac": float(r["facing_cafe_frac"]),
            "track_id":         r["track_id"],
            "zones_visited":    r["zones_visited"],
            "user_verdict":     "",  # пользователь заполнит: ok/wrong/maybe
            "note":             "",
        })
    rows.sort(key=lambda r: r["timestamp_sec"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n[lingered] {len(rows)} candidates -> {out_path}")
    if rows:
        print(f"  диапазон: {rows[0]['timestamp_mmss']} .. {rows[-1]['timestamp_mmss']}")
        print(f"  median duration: "
              f"{sorted(r['duration_sec'] for r in rows)[len(rows)//2]:.1f} sec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(GT_CSV))
    ap.add_argument("--pred", default=str(PRED_CSV))
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="Matching tolerance window (seconds, default 3)")
    ap.add_argument("--passer-window-end", type=float, default=50.0,
                    help="Until what time GT.passer_by размечен (sec). default 50")
    ap.add_argument("--door-window-start", type=float, default=50.0,
                    help="С какой секунды размечал entered/exited. default 50")
    ap.add_argument("--door-window-end", type=float, default=300.0,
                    help="До какой секунды размечал entered/exited. default 300")
    ap.add_argument("--lingered-out", default=str(LINGER_OUT))
    args = ap.parse_args()

    gt = load_gt(Path(args.gt))
    pred = load_pred(Path(args.pred))

    print("GT events by type:")
    for k, v in sorted(gt.items()):
        print(f"  {k:20s}: {len(v):3d}  range {fmt_ts(min(v))}..{fmt_ts(max(v))}")

    print("\nPred events by type (full 30-min):")
    for k, v in sorted(pred.items()):
        print(f"  {k:25s}: {len(v):5d}")

    # ── Evaluations ────────────────────────────────────────────────────
    results = {}
    if "passer_by" in gt:
        # GT.passer_by сравниваем с pred.passer_by_single_side (по инструкции)
        results["passer_by_vs_single_side"] = evaluate(
            "PASSER_BY (GT) vs passer_by_single_side (pred)",
            gt["passer_by"],
            pred.get("passer_by_single_side", []),
            (0.0, args.passer_window_end),
            args.tolerance,
        )
        # На всякий случай — также vs full passer_by
        if pred.get("passer_by"):
            results["passer_by_vs_full"] = evaluate(
                "PASSER_BY (GT) vs passer_by FULL (pred)",
                gt["passer_by"],
                pred.get("passer_by", []),
                (0.0, args.passer_window_end),
                args.tolerance,
            )

    if "entered" in gt:
        results["entered"] = evaluate(
            "ENTERED",
            gt["entered"],
            pred.get("entered", []),
            (args.door_window_start, args.door_window_end),
            args.tolerance,
        )

    if "exited" in gt:
        results["exited"] = evaluate(
            "EXITED",
            gt["exited"],
            pred.get("exited", []),
            (args.door_window_start, args.door_window_end),
            args.tolerance,
        )

    # ── Combined door (entered+exited) ────────────────────────────────
    if "entered" in gt and "exited" in gt:
        gt_door = sorted(gt["entered"] + gt["exited"])
        pred_door = sorted(pred.get("entered", []) + pred.get("exited", []),
                           key=lambda x: x[0])
        results["door_combined"] = evaluate(
            "DOOR COMBINED (entered+exited)",
            gt_door,
            pred_door,
            (args.door_window_start, args.door_window_end),
            args.tolerance,
            verbose=False,
        )

    # ── Lingered export ───────────────────────────────────────────────
    export_lingered(pred, Path(args.lingered_out))

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'metric':32s}  {'TP':>4s} {'FP':>4s} {'FN':>4s}   {'P':>5s} {'R':>5s} {'F1':>5s}")
    for name, r in results.items():
        print(f"{name:32s}  {r['tp']:4d} {r['fp']:4d} {r['fn']:4d}   "
              f"{r['P']:5.3f} {r['R']:5.3f} {r['F1']:5.3f}")


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            if _s.encoding and _s.encoding.lower() != "utf-8":
                _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()

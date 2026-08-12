"""
Честный замер производительности пайплайна.

Зачем отдельный скрипт. «140 fps на такой-то карте» — самая часто завышаемая
цифра в CV-портфолио, потому что обычно меряют только forward pass модели и
выдают его за пропускную способность системы. На собеседовании про это
спрашивают прицельно: «это чистый инференс или с декодом видео,
препроцессингом и NMS?»

Скрипт меряет стадии по нарастающей, чтобы было видно цену каждой:

    decode           только чтение и декодирование кадров
    + detect         детекция (включая letterbox и NMS внутри ultralytics)
    + track          трекинг ByteTrack поверх детекций
    + count          обновление счётчика — полный пайплайн

Разница между соседними строками и есть стоимость стадии. Итоговая цифра для
резюме — последняя строка, а не первая.

Запуск:
    python scripts/benchmark_fps.py --source video.mp4 --model yolo11l.pt \
        --device 0 --imgsz 640 --frames 300
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footfall.counters import SingleLineCounter          # noqa: E402
from footfall.detect import Detector                     # noqa: E402
from footfall.sources import open_source                 # noqa: E402
from footfall.track import Tracker                       # noqa: E402


def describe_device(device: str | None) -> str:
    """Что именно считало. Без этого цифра бессмысленна."""
    try:
        import torch
        if device and device != "cpu" and torch.cuda.is_available():
            idx = int(device) if device.isdigit() else 0
            name = torch.cuda.get_device_name(idx)
            return f"{name} (CUDA)"
        return f"{platform.processor() or 'CPU'} (CPU)"
    except ImportError:
        return device or "unknown"


def read_frames(cap, n: int) -> list:
    """Прочитать n кадров в память — чтобы декод не мешал замеру стадий."""
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def bench(label: str, fn, frames: list, warmup: int) -> tuple[str, float, float]:
    """Прогнать fn по кадрам. Первые warmup кадров не считаются.

    Разогрев обязателен: первый вызов модели тянет за собой аллокацию памяти
    на устройстве, компиляцию ядер CUDA и подгрузку весов в кэш. Если его
    включить в замер, цифра занижается тем сильнее, чем короче прогон.
    """
    for f in frames[:warmup]:
        fn(f)

    measured = frames[warmup:]
    if not measured:
        raise SystemExit("кадров меньше, чем нужно на разогрев")

    t0 = time.perf_counter()
    for f in measured:
        fn(f)
    elapsed = time.perf_counter() - t0

    fps = len(measured) / elapsed
    ms = elapsed / len(measured) * 1000
    return label, fps, ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolo11l.pt")
    ap.add_argument("--device", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--half", action="store_true",
                    help="FP16. Указывать в отчёте обязательно — цифры FP16 "
                         "и FP32 несопоставимы")
    ap.add_argument("--frames", type=int, default=300,
                    help="сколько кадров участвует в замере")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    device = args.device or None

    # ── декод отдельно: он не зависит от модели и часто оказывается
    #    узким местом на 4K-потоке
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")

    t0 = time.perf_counter()
    frames = read_frames(cap, args.frames + args.warmup)
    decode_elapsed = time.perf_counter() - t0
    cap.release()

    if len(frames) <= args.warmup:
        raise SystemExit(f"в источнике только {len(frames)} кадров")

    h, w = frames[0].shape[:2]
    decode_fps = len(frames) / decode_elapsed

    detector = Detector(args.model, device=device, conf=args.conf,
                        imgsz=args.imgsz, half=args.half)
    tracker = Tracker()
    counter = SingleLineCounter(sv.Point(0, h // 2), sv.Point(w, h // 2),
                                sv.Position.BOTTOM_CENTER, min_cross=2)

    def stage_detect(frame):
        return detector.detect(frame)

    def stage_track(frame):
        return tracker.update(detector.detect(frame))

    def stage_full(frame):
        counter.update(tracker.update(detector.detect(frame)))

    rows = [("decode only", decode_fps, 1000 / decode_fps)]
    rows.append(bench("+ detect", stage_detect, frames, args.warmup))
    tracker.reset()
    rows.append(bench("+ track", stage_track, frames, args.warmup))
    tracker.reset()
    rows.append(bench("+ count (полный пайплайн)", stage_full, frames, args.warmup))

    print()
    print("КОНФИГУРАЦИЯ ЗАМЕРА")
    print(f"  устройство   {describe_device(device)}")
    print(f"  модель       {Path(args.model).name}")
    print(f"  разрешение   {w}x{h} -> imgsz {args.imgsz}")
    print(f"  точность     {'FP16' if args.half else 'FP32'}")
    print(f"  батч         1 (потоковая обработка, кадр за кадром)")
    print(f"  кадров       {len(frames) - args.warmup} (+{args.warmup} на разогрев)")
    print()
    print(f"  {'стадия':<28} {'fps':>9} {'мс/кадр':>10}")
    print("  " + "-" * 49)
    for label, fps, ms in rows:
        print(f"  {label:<28} {fps:>9.1f} {ms:>10.2f}")
    print()
    print("  Для резюме и README берётся ПОСЛЕДНЯЯ строка — полный пайплайн.")
    print("  Строка '+ detect' это только инференс; выдавать её за пропускную")
    print("  способность системы некорректно.")
    print()
    print("  Декод измерен на чтении в память и в реальном прогоне идёт")
    print("  параллельно с остальным, поэтому сквозная цифра обычно выше,")
    print("  чем 1/(сумма стадий).")


if __name__ == "__main__":
    main()

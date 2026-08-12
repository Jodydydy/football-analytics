"""
Геометрия сцены: линии и полигоны зон.

Полигоны рисуются один раз на установке камеры и лежат в JSON. Это ручная
работа на каждой точке — главная скрытая стоимость внедрения: десять столов
это десять полигонов, а если мебель переставили, всё пересчитывается заново.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .sources import parse_source


def parse_xy(s: str) -> tuple[int, int]:
    """Разобрать 'x,y' в пару целых."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'x,y', got {s!r}")
    return int(parts[0]), int(parts[1])


def load_zone(path: str) -> np.ndarray:
    """Прочитать полигон из JSON. Вернуть массив (N, 2), N >= 3."""
    data = json.loads(Path(path).read_text())
    poly = np.array(data["polygon"], dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise SystemExit(f"{path}: polygon must be Nx2 with N>=3, got {poly.shape}")
    return poly


def load_zones(paths: str) -> list[np.ndarray]:
    """Несколько полигонов из списка путей через запятую."""
    return [load_zone(p.strip()) for p in paths.split(",") if p.strip()]


def preview_line(source: str, y_frac: float,
                 start: str | None, end: str | None,
                 out: str = "preview_line.jpg") -> None:
    """Сохранить первый кадр с нарисованной линией.

    Нужно перед прогоном: линия, выбранная «на глаз» по описанию сцены, почти
    всегда оказывается не там. Дешевле посмотреть один кадр, чем прогнать
    четыре часа видео и обнаружить, что считались проходящие мимо.

    Зелёная точка — start, синяя — end. Порядок концов задаёт полярность
    IN/OUT: если счётчик перепутал направления, концы меняются местами.
    """
    cap = cv2.VideoCapture(parse_source(source))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame from {source}")

    h, w = frame.shape[:2]
    if start is not None and end is not None:
        sx, sy = parse_xy(start)
        ex, ey = parse_xy(end)
        cv2.line(frame, (sx, sy), (ex, ey), (0, 0, 255), 3)
        cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (ex, ey), 6, (255, 0, 0), -1)
        label = f"line ({sx},{sy})->({ex},{ey})  ({w}x{h})"
    else:
        ly = int(h * y_frac)
        cv2.line(frame, (0, ly), (w, ly), (0, 0, 255), 3)
        label = f"line y={y_frac:.2f}  ({w}x{h})"

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(out, frame)
    print(f"wrote {out} — open it, adjust, rerun if needed")

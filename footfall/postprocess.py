"""
Постобработка лога событий.

Два фильтра, оба чинят одну и ту же болезнь — дрожание рамки детектора у
самой линии, — но разные её проявления. Оба нашлись на реальных данных, а не
придуманы заранее.

Почему постобработка, а не фильтрация на лету: чтобы увидеть, что событие
ложное, нужно знать будущее. Пара «вошёл — вышел» за полсекунды выглядит
ложной только когда пришло второе событие. Онлайн так нельзя, поэтому чистка
идёт по готовому логу.

Цена решения: в реальном времени события уезжают в лог как есть, и алерты
могут сработать на мусоре. Для отчётности это неважно, для realtime-уведомлений
пришлось бы держать буфер и задерживать выдачу.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_split(log_path: Path, pre_session_lines: int):
    """Прочитать лог и отделить строки текущей сессии от предыдущих."""
    with log_path.open("r", newline="") as f:
        all_rows = list(csv.reader(f))
    if pre_session_lines >= len(all_rows):
        return None, None
    return all_rows[:pre_session_lines], all_rows[pre_session_lines:]


def _index_by_track(session: list[list[str]]) -> dict[str, list[int]]:
    """track_id -> номера строк с событиями in/out."""
    by_track: dict[str, list[int]] = {}
    for idx, row in enumerate(session):
        if len(row) < 4 or row[2] not in ("in", "out"):
            continue
        by_track.setdefault(row[3], []).append(idx)
    return by_track


def _rewrite(log_path: Path, head, session, drop: set[int]) -> int:
    if not drop:
        return 0
    kept = head + [r for i, r in enumerate(session) if i not in drop]
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerows(kept)
    return len(drop)


def cancel_roundtrips(log_path: Path, pre_session_lines: int,
                      cancel_window: int) -> int:
    """Выбросить пары IN<->OUT одного трека, случившиеся за cancel_window кадров.

    Симптом: человек стоит вплотную к линии, рамка детектора подрагивает,
    якорная точка перескакивает через линию туда и обратно. Получается пара
    противоположных событий с интервалом в несколько кадров. Настоящий проход
    так не выглядит: развернуться и уйти за полсекунды невозможно.

    Правит файл на месте, трогает только строки текущей сессии.
    Возвращает число выброшенных строк.
    """
    head, session = _read_split(log_path, pre_session_lines)
    if head is None:
        return 0

    by_track = _index_by_track(session)
    drop: set[int] = set()
    for idxs in by_track.values():
        idxs = [i for i in idxs if i not in drop]
        idxs.sort(key=lambda i: int(session[i][0]))
        i = 0
        while i + 1 < len(idxs):
            a, b = idxs[i], idxs[i + 1]
            frame_a, frame_b = int(session[a][0]), int(session[b][0])
            event_a, event_b = session[a][2], session[b][2]
            if event_a != event_b and (frame_b - frame_a) <= cancel_window:
                drop.add(a)
                drop.add(b)
                i += 2
            else:
                i += 1

    return _rewrite(log_path, head, session, drop)


def apply_cooldown(log_path: Path, pre_session_lines: int,
                   cooldown_frames: int) -> int:
    """Выбросить повторы В ТУ ЖЕ СТОРОНУ по одному треку внутри cooldown_frames.

    Дополняет cancel_roundtrips, который убирает только противоположные пары.
    Здесь лечится другой случай: трек застрял на линии и выдал серию событий
    одного направления. На CAVIAR, последовательность OneStopEnter2cor, один
    трек дал семь событий OUT с интервалами 13-65 кадров — окно отмены в
    8 кадров до них не дотягивалось, потому что пары были однонаправленные.

    Правит файл на месте. Возвращает число выброшенных строк.
    """
    head, session = _read_split(log_path, pre_session_lines)
    if head is None:
        return 0

    by_track = _index_by_track(session)
    drop: set[int] = set()
    for idxs in by_track.values():
        idxs = sorted(idxs, key=lambda i: int(session[i][0]))
        last_frame_per_dir: dict[str, int] = {}
        for i in idxs:
            frame = int(session[i][0])
            direction = session[i][2]
            last = last_frame_per_dir.get(direction)
            if last is not None and (frame - last) < cooldown_frames:
                drop.add(i)
            else:
                last_frame_per_dir[direction] = frame

    return _rewrite(log_path, head, session, drop)


def count_directions(log_path: Path, pre_session_lines: int) -> tuple[int, int]:
    """Пересчитать события in и out в текущей сессии — уже после чистки."""
    with log_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    in_n = out_n = 0
    for row in rows[pre_session_lines:]:
        if len(row) < 4:
            continue
        if row[2] == "in":
            in_n += 1
        elif row[2] == "out":
            out_n += 1
    return in_n, out_n

"""
Счётчики событий: превращают треки в события «вошёл», «вышел», «забрал заказ»,
«очередь», «на рабочем месте никого».

Четыре стратегии, каждая под свою геометрию:

    SingleLineCounter   одна линия, зачёт при пересечении
    TripwireCounter     пара параллельных линий, зачёт только при пересечении обеих
    ZoneDwellCounter    полигон + таймер: засчитать того, кто задержался
    ZoneAbsenceCounter  полигон наоборот: тревога, когда зона пуста слишком долго

Общая механика: на каждом кадре в update() приходят детекции с tracker_id,
на выходе — списки track_id, по которым событие произошло ИМЕННО СЕЙЧАС.
Счётчик хранит состояние между кадрами, поэтому один объект живёт на всё видео.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


def anchor_points(detections: sv.Detections,
                  anchor: sv.Position) -> np.ndarray:
    """Точка бокса, по которой решается «внутри или снаружи».

    Выбор точки — не мелочь, а решение с последствиями:

      BOTTOM_CENTER  ноги. Правильно для камеры под углом: человек стоит
                     на полу, и зона нарисована по полу. Ломается, когда
                     ноги не видны (загорожены прилавком, обрезаны кадром).
      CENTER         центр бокса. Устойчивее к обрезанию, но при виде под
                     углом центр «висит в воздухе» и заходит в зону раньше ног.
      TOP_CENTER     голова. Работает на плотной толпе, где ноги не видны
                     почти никогда, и на камерах сверху.

    Возвращает массив (N, 2) координат.
    """
    x1, y1, x2, y2 = (detections.xyxy[:, 0], detections.xyxy[:, 1],
                      detections.xyxy[:, 2], detections.xyxy[:, 3])
    cx = (x1 + x2) / 2
    if anchor == sv.Position.BOTTOM_CENTER:
        cy = y2
    elif anchor == sv.Position.TOP_CENTER:
        cy = y1
    else:
        cy = (y1 + y2) / 2
    return np.column_stack([cx, cy])


class SingleLineCounter:
    """Одна линия с дедупликацией по треку.

    Каждый track_id засчитывается не более одного раза в каждую сторону.
    Без этого человек, стоящий вплотную к линии, пересчитывается на каждом
    микродвижении: дрожание рамки детектора туда-сюда даёт десятки событий
    от одного человека. Именно это ломало подсчёт в очереди у прилавка.
    """

    def __init__(self, start: sv.Point, end: sv.Point,
                 anchor: sv.Position, min_cross: int, dedup: bool = True):
        self.line = sv.LineZone(
            start=start, end=end,
            triggering_anchors=[anchor],
            minimum_crossing_threshold=min_cross,
        )
        self._in_count = 0
        self._out_count = 0
        self.dedup = dedup
        self.counted_in: set[int] = set()
        self.counted_out: set[int] = set()

    @property
    def in_count(self) -> int:
        return self._in_count

    @property
    def out_count(self) -> int:
        return self._out_count

    def update(self, detections: sv.Detections) -> tuple[list[int], list[int]]:
        crossed_in, crossed_out = self.line.trigger(detections)
        in_tids: list[int] = []
        out_tids: list[int] = []
        if detections.tracker_id is None:
            return in_tids, out_tids
        for i, raw_tid in enumerate(detections.tracker_id):
            tid = int(raw_tid)
            if crossed_in[i] and (not self.dedup or tid not in self.counted_in):
                in_tids.append(tid)
                self.counted_in.add(tid)
                self._in_count += 1
            if crossed_out[i] and (not self.dedup or tid not in self.counted_out):
                out_tids.append(tid)
                self.counted_out.add(tid)
                self._out_count += 1
        return in_tids, out_tids

    def annotate(self, frame, line_annot: sv.LineZoneAnnotator):
        return line_annot.annotate(frame, self.line)


class TripwireCounter:
    """Пара параллельных линий по обе стороны от базовой.

    Событие засчитывается, только если трек пересёк ОБЕ линии в одном
    направлении не более чем за max_gap_frames кадров.

    Зачем: одиночная линия срабатывает на любом касании. Человек, постоявший
    в дверях и вернувшийся, даёт ложное событие. Пара линий требует
    завершённого прохода — «зашёл с одной стороны, вышел с другой».
    Это заметно чище, но теряет тех, кто прошёл слишком быстро (обе линии
    в одном кадре) или слишком медленно (разрыв больше max_gap_frames).

    Направление IN/OUT наследуется от sv.LineZone и зависит от порядка
    start->end базовой линии. Если полярность перепутана — поменять концы
    местами.
    """

    def __init__(self, start: sv.Point, end: sv.Point, delta: float,
                 anchor: sv.Position, min_cross: int, max_gap_frames: int):
        dx = end.x - start.x
        dy = end.y - start.y
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1.0:
            raise ValueError(f"line is degenerate: start={start} end={end}")
        # единичная нормаль, повёрнутая на 90° против часовой стрелки
        nx, ny = -dy / length, dx / length
        sa = sv.Point(int(round(start.x + nx * delta)), int(round(start.y + ny * delta)))
        ea = sv.Point(int(round(end.x + nx * delta)), int(round(end.y + ny * delta)))
        sb = sv.Point(int(round(start.x - nx * delta)), int(round(start.y - ny * delta)))
        eb = sv.Point(int(round(end.x - nx * delta)), int(round(end.y - ny * delta)))
        self.line_a = sv.LineZone(start=sa, end=ea, triggering_anchors=[anchor],
                                  minimum_crossing_threshold=min_cross)
        self.line_b = sv.LineZone(start=sb, end=eb, triggering_anchors=[anchor],
                                  minimum_crossing_threshold=min_cross)
        self.max_gap = max_gap_frames
        self.tracks: dict[int, dict] = {}
        self.in_count = 0
        self.out_count = 0
        self.counted_in: set[int] = set()
        self.counted_out: set[int] = set()
        self._frame = 0

    def update(self, detections: sv.Detections) -> tuple[list[int], list[int]]:
        self._frame += 1
        a_in, a_out = self.line_a.trigger(detections)
        b_in, b_out = self.line_b.trigger(detections)
        completed_in: list[int] = []
        completed_out: list[int] = []
        if detections.tracker_id is None:
            return completed_in, completed_out
        for i, raw_tid in enumerate(detections.tracker_id):
            tid = int(raw_tid)
            st = self.tracks.setdefault(tid, {
                "a_dir": None, "a_frame": -1,
                "b_dir": None, "b_frame": -1,
            })
            if a_in[i]:
                st["a_dir"], st["a_frame"] = "in", self._frame
            elif a_out[i]:
                st["a_dir"], st["a_frame"] = "out", self._frame
            if b_in[i]:
                st["b_dir"], st["b_frame"] = "in", self._frame
            elif b_out[i]:
                st["b_dir"], st["b_frame"] = "out", self._frame

            if (st["a_dir"] == "in" and st["b_dir"] == "in"
                    and abs(st["a_frame"] - st["b_frame"]) <= self.max_gap):
                if tid not in self.counted_in:
                    self.in_count += 1
                    completed_in.append(tid)
                    self.counted_in.add(tid)
                st["a_dir"] = st["b_dir"] = None
            elif (st["a_dir"] == "out" and st["b_dir"] == "out"
                    and abs(st["a_frame"] - st["b_frame"]) <= self.max_gap):
                if tid not in self.counted_out:
                    self.out_count += 1
                    completed_out.append(tid)
                    self.counted_out.add(tid)
                st["a_dir"] = st["b_dir"] = None

        return completed_in, completed_out

    def annotate(self, frame, line_annot: sv.LineZoneAnnotator):
        frame = line_annot.annotate(frame, self.line_a)
        frame = line_annot.annotate(frame, self.line_b)
        return frame


class ZoneDwellCounter:
    """Полигон с таймером: засчитать трек, пробывший в зоне >= dwell_frames.

    Задача — считать транзакции на стойке выдачи. Клиент заходит в зону,
    ждёт несколько секунд, уходит. Прохожий, срезающий через зону, не
    накапливает достаточно кадров и не засчитывается. Именно таймер
    отличает «забрал заказ» от «прошёл мимо».

    Режимы подсчёта
    ---------------
    dwell-unique  каждый уникальный track_id считается один раз. Точно, пока
                  треки стабильны. В плотной группе рассыпается: один человек
                  с двумя ID даёт два события.

    session-max   сессия = цикл «зона пуста -> непуста -> снова пуста». За
                  сессию засчитывается ПИКОВОЕ число одновременных посетителей.
                  Невосприимчиво к ID switch, потому что уникальные ID вообще
                  не используются. Плата: двое подряд по одному дадут 1, а не 2.

    hybrid-max    максимум из двух предыдущих: пик одновременных ИЛИ число
                  уникальных треков, набравших dwell. Компромисс.

    Выбор режима — не вкусовщина, а следствие плотности сцены. На разреженном
    входе выигрывает dwell-unique, на плотной выдаче — session-max.
    """

    def __init__(self, polygon: np.ndarray, anchor: sv.Position,
                 dwell_frames: int,
                 queue_threshold: int = 0,
                 queue_alert_frames: int = 0,
                 count_mode: str = "dwell-unique",
                 session_min_frames: int = 0):
        self.polygon = polygon.astype(np.int32)
        self.anchor = anchor
        self.dwell_frames = dwell_frames
        self.dwell: dict[int, int] = {}
        self.counted: set[int] = set()
        self._in_count = 0

        # очередь: тревога при устойчивом превышении порога (0 = выключено)
        self.queue_threshold = queue_threshold
        self.queue_alert_frames = queue_alert_frames
        self._above_for = 0
        self._alert_armed = True
        self._current_occupants = 0
        self._pending_alert: int | None = None

        # состояние сессии для session-max / hybrid-max
        self.count_mode = count_mode
        self.session_min_frames = session_min_frames
        self._session_max = 0
        self._session_frames = 0
        self._session_idx = 0
        self._session_tids: set[int] = set()

    @property
    def in_count(self) -> int:
        return self._in_count

    @property
    def out_count(self) -> int:
        return 0

    @property
    def current_occupants(self) -> int:
        return self._current_occupants

    def update(self, detections: sv.Detections) -> tuple[list[int], list[int]]:
        in_tids: list[int] = []
        current = 0

        if detections.tracker_id is not None and len(detections) > 0:
            anchors = anchor_points(detections, self.anchor)
            for i, raw_tid in enumerate(detections.tracker_id):
                tid = int(raw_tid)
                x, y = float(anchors[i, 0]), float(anchors[i, 1])
                inside = cv2.pointPolygonTest(self.polygon, (x, y), False) >= 0
                if inside:
                    current += 1
                    self.dwell[tid] = self.dwell.get(tid, 0) + 1
                    self._session_tids.add(tid)
                    if (self.count_mode == "dwell-unique"
                            and self.dwell[tid] >= self.dwell_frames
                            and tid not in self.counted):
                        self.counted.add(tid)
                        self._in_count += 1
                        in_tids.append(tid)

        self._current_occupants = current

        if self.count_mode in ("session-max", "hybrid-max"):
            in_tids.extend(self._update_session(current))

        if self.queue_threshold > 0:
            self._update_queue_alert(current)

        return in_tids, []

    def _update_session(self, current: int) -> list[int]:
        """Учёт сессий. Возвращает синтетические id, если сессия закрылась."""
        if current > 0:
            self._session_frames += 1
            self._session_max = max(self._session_max, current)
            return []

        emitted: list[int] = []
        if (self._session_max > 0
                and self._session_frames >= self.session_min_frames):
            if self.count_mode == "session-max":
                emit = self._session_max
            else:  # hybrid-max
                unique = sum(1 for t in self._session_tids
                             if self.dwell.get(t, 0) >= self.dwell_frames)
                emit = max(self._session_max, unique)
            self._in_count += emit
            # синтетические id: реальных треков у сессии нет, но лог событий
            # ждёт идентификатор в каждой строке
            base = 100_000 + self._session_idx * 100
            emitted = [base + i for i in range(emit)]
            self._session_idx += 1

        self._session_max = 0
        self._session_frames = 0
        # self.dwell намеренно НЕ сбрасывается: трек, который вышел и вернулся,
        # сохраняет накопленное время
        self._session_tids = set()
        return emitted

    def _update_queue_alert(self, current: int) -> None:
        """Тревога при устойчивом превышении порога, со взводом после спада."""
        if current >= self.queue_threshold:
            self._above_for += 1
            if self._above_for >= self.queue_alert_frames and self._alert_armed:
                self._alert_armed = False
                self._pending_alert = current
        else:
            self._above_for = 0
            self._alert_armed = True

    def pop_queue_alert(self) -> int | None:
        """Число людей в зоне, если тревога сработала на этом кадре. Затем сброс."""
        a = self._pending_alert
        self._pending_alert = None
        return a

    def annotate(self, frame, detections: sv.Detections, fps: float) -> np.ndarray:
        pts = self.polygon.reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 165, 0), thickness=2)
        if detections.tracker_id is None or len(detections) == 0:
            return frame
        anchors = anchor_points(detections, self.anchor)
        for i, raw_tid in enumerate(detections.tracker_id):
            tid = int(raw_tid)
            n = self.dwell.get(tid, 0)
            if n == 0:
                continue
            x, y = int(anchors[i, 0]), int(anchors[i, 1])
            color = (0, 200, 0) if tid in self.counted else (0, 200, 255)
            cv2.putText(frame, f"#{tid} {n / fps:.1f}s", (x - 30, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


class ZoneAbsenceCounter:
    """Тревога, когда зона пуста дольше absence_frames подряд.

    Задача обратная ZoneDwellCounter: не «кто-то пришёл», а «никого нет».
    Применение — сотрудник ушёл с рабочего места. Полигон рисуется по зоне,
    где сотрудник должен находиться (за стойкой), причём геометрически туда
    не может зайти посетитель — это самый дешёвый способ отличить одного
    от другого, без классификации по одежде и без распознавания лиц.

    Про fire_from_start
    -------------------
    По умолчанию первая тревога срабатывает только после того, как зона хотя бы
    раз была занята. Защита от холодного старта: детектор в первых кадрах ещё
    не разогрет, зона читается пустой, летит ложная тревога.

    Но у защиты есть цена: если запись начинается с уже пустого рабочего места,
    настоящее событие «смена не началась» подавляется. На датасете Edinburgh дни
    4, 6 и 9 потеряли по реальному событию именно так.

    fire_from_start=True снимает защиту — уместно, когда запись гарантированно
    начинается с сотрудником на месте, либо когда отсутствие с начала смены
    само по себе является событием.

    Тонкость реализации: здесь НЕ требуется tracker_id. Для факта присутствия
    личность неважна, а на источниках с низким fps ByteTrack часто ещё не успел
    активировать трек и отдаёт tracker_id=None — зона читалась бы пустой и
    сыпала ложными тревогами.
    """

    def __init__(self, polygon: np.ndarray, anchor: sv.Position,
                 absence_frames: int, fire_from_start: bool = False):
        self.polygon = polygon.astype(np.int32)
        self.anchor = anchor
        self.absence_frames = absence_frames
        self._empty_for = 0
        self._alert_armed = True
        self._was_ever_occupied = fire_from_start
        self._currently_occupied = False

    @property
    def currently_occupied(self) -> bool:
        return self._currently_occupied

    @property
    def empty_frames(self) -> int:
        return self._empty_for

    def update(self, detections: sv.Detections) -> int | None:
        """Длительность отсутствия в кадрах, если тревога сработала. Иначе None."""
        someone_inside = False
        if len(detections) > 0:
            anchors = anchor_points(detections, self.anchor)
            for i in range(len(detections)):
                x, y = float(anchors[i, 0]), float(anchors[i, 1])
                if cv2.pointPolygonTest(self.polygon, (x, y), False) >= 0:
                    someone_inside = True
                    break
        self._currently_occupied = someone_inside

        if someone_inside:
            self._was_ever_occupied = True
            self._empty_for = 0
            self._alert_armed = True
            return None
        if not self._was_ever_occupied:
            return None
        self._empty_for += 1
        if self._empty_for >= self.absence_frames and self._alert_armed:
            self._alert_armed = False
            return self._empty_for
        return None

    def annotate(self, frame, fps: float) -> np.ndarray:
        pts = self.polygon.reshape(-1, 1, 2)
        color = (0, 200, 0) if self._currently_occupied else (0, 80, 220)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        if not self._currently_occupied and self._was_ever_occupied:
            x = int(self.polygon[:, 0].mean())
            y = int(self.polygon[:, 1].min()) - 8
            cv2.putText(frame, f"empty {self._empty_for / fps:.0f}s", (x - 40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return frame

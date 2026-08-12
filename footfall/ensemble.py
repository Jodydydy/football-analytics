"""
Ансамбль детекторов: тело + голова.

Задача: у прилавка и в очереди тело человека закрыто, а голова видна.
Детектор людей такого не находит — рекол проседает ровно там, где считать
важнее всего. Отдельная модель детекции голов находит, но её рамки нельзя
подать в счётчик напрямую: зоны и линии нарисованы под ростовые рамки.

Решение: голову достраивают до синтетической ростовой рамки по анатомической
пропорции, и добавляют, только если на этом месте ещё нет настоящей детекции.

Ограничение, которое надо называть самому: пропорция «голова это 1/7 роста»
верна для взрослого человека в полный рост анфас. Для сидящего, для ребёнка,
для сильного ракурса сверху она врёт. Синтетическая рамка нужна не для
измерения, а для попадания якорной точки в правильную зону — и для этой цели
приблизительной пропорции хватает.
"""
from __future__ import annotations

import numpy as np
import supervision as sv


def head_to_body_bbox(head_xyxy: np.ndarray) -> np.ndarray:
    """Достроить рамки голов до синтетических ростовых.

    Пропорции: высота тела = 7 высот головы, ширина = 2 ширины головы.
    Голова помещается сверху построенного тела.
    """
    if len(head_xyxy) == 0:
        return head_xyxy.astype(np.float32)

    hx1, hy1, hx2, hy2 = (head_xyxy[:, 0], head_xyxy[:, 1],
                          head_xyxy[:, 2], head_xyxy[:, 3])
    cx = (hx1 + hx2) / 2
    body_w = (hx2 - hx1) * 2
    body_h = np.maximum(hy2 - hy1, 1) * 7  # maximum, чтобы не делить на ноль

    return np.column_stack([
        cx - body_w / 2,
        hy1,
        cx + body_w / 2,
        hy1 + body_h,
    ]).astype(np.float32)


def _iou_against(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU одной рамки против массива рамок. Векторно, без цикла."""
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_box + area_others - inter + 1e-6)


def merge_person_and_head(person_dets: sv.Detections,
                          head_dets: sv.Detections | None,
                          iou_drop: float = 0.3) -> sv.Detections:
    """Слить детекции тел и достроенные из голов.

    Синтетическая рамка отбрасывается, если пересекается с любой настоящей
    детекцией человека сильнее iou_drop — иначе один человек попал бы в
    счётчик дважды.
    """
    if head_dets is None or len(head_dets) == 0:
        return person_dets

    synth = head_to_body_bbox(head_dets.xyxy)
    head_conf = head_dets.confidence

    if len(person_dets) > 0:
        keep = [i for i in range(len(synth))
                if _iou_against(synth[i], person_dets.xyxy).max() <= iou_drop]
        if not keep:
            return person_dets
        synth = synth[keep]
        head_conf = head_conf[keep] if head_conf is not None else None

    conf_arr = (head_conf if head_conf is not None
                else np.full(len(synth), 0.5, dtype=np.float32))

    if len(person_dets) > 0:
        person_conf = (person_dets.confidence if person_dets.confidence is not None
                       else np.ones(len(person_dets), dtype=np.float32))
        person_cls = (person_dets.class_id if person_dets.class_id is not None
                      else np.zeros(len(person_dets), dtype=int))
        xyxy = np.concatenate([person_dets.xyxy, synth])
        conf = np.concatenate([person_conf, conf_arr])
        cls = np.concatenate([person_cls, np.zeros(len(synth), dtype=int)])
    else:
        xyxy, conf, cls = synth, conf_arr, np.zeros(len(synth), dtype=int)

    # Собираем sv.Detections из сырых массивов, а не через sv.Detections.merge():
    # merge требует одинаковых ключей в .data, а у двух разных моделей словари
    # с метаданными детекций не совпадают
    return sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=conf.astype(np.float32),
        class_id=cls.astype(int),
    )

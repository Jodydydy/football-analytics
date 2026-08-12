"""
Командный интерфейс.

    python -m footfall.cli --source video.mp4 --show
    python -m footfall.cli --source video.mp4 --line 0.55 --preview-line
    python -m footfall.cli --source video.mp4 --zone pickup.json --show

Аргументы сгруппированы по смыслу: источник и модель, геометрия зачёта,
детекция, трекинг, режим зоны, тревоги, вывод.
"""
from __future__ import annotations

import argparse

from .pipeline import run
from .zones import preview_line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="footfall",
        description="Подсчёт посетителей и событий по видео с камеры.")

    src = ap.add_argument_group("источник и модель")
    src.add_argument("--source", required=True,
                     help="видеофайл, индекс вебкамеры (0), RTSP URL или папка с кадрами")
    src.add_argument("--model", default="yolo11l.pt",
                     help="веса детектора. n/s/m/l/x — от быстрых к точным")
    src.add_argument("--device", default="",
                     help="'0' для первой CUDA-карты, 'cpu', пусто = автоопределение")

    geo = ap.add_argument_group("геометрия зачёта")
    geo.add_argument("--line", type=float, default=0.5,
                     help="доля высоты кадра (0..1) для горизонтальной линии")
    geo.add_argument("--start", help="начало линии 'x,y' в пикселях (вместе с --end)")
    geo.add_argument("--end", help="конец линии 'x,y' в пикселях (вместе с --start)")
    geo.add_argument("--anchor", default="bottom-center",
                     choices=["bottom-center", "center", "top-center"],
                     help="точка рамки, по которой проверяется пересечение")
    geo.add_argument("--min-cross", type=int, default=2,
                     help="сколько кадров трек должен продержаться на новой стороне")
    geo.add_argument("--gap", type=float, default=0.0,
                     help="разнос пары линий как доля высоты кадра; 0 = одна линия")
    geo.add_argument("--max-gap-frames", type=int, default=30,
                     help="максимум кадров между пересечениями A и B")
    geo.add_argument("--invert-direction", action="store_true",
                     help="поменять IN и OUT местами. Полярность sv.LineZone "
                          "зависит от направления start->end; проще перевернуть "
                          "флагом, чем перерисовывать линию")

    det = ap.add_argument_group("детекция")
    det.add_argument("--conf", type=float, default=0.35, help="порог уверенности")
    det.add_argument("--imgsz", type=int, default=640,
                     help="размер входа; больше = лучше видно мелких, медленнее")
    det.add_argument("--tta", action="store_true",
                     help="test-time augmentation: +3-5 п.п. рекола, ~2x времени")
    det.add_argument("--preproc", default="none", choices=["none", "clahe"],
                     help="clahe: выравнивание яркости. Помогает против "
                          "контрового света от входной двери")
    det.add_argument("--ensemble-head", default=None,
                     help="веса детектора голов. Каждая голова достраивается "
                          "до синтетической ростовой рамки и добавляется, если "
                          "на этом месте нет настоящей детекции. Поднимает "
                          "рекол на закрытых прилавком телах")
    det.add_argument("--ensemble-iou", type=float, default=0.3,
                     help="выше этого IoU синтетическая рамка считается дублем")

    trk = ap.add_argument_group("трекинг")
    trk.add_argument("--track-buffer", type=int, default=30,
                     help="сколько кадров трек живёт после потери объекта. "
                          "Больше = меньше ID switch в толпе, но больше "
                          "фантомных треков от уже ушедших людей")
    trk.add_argument("--track-min-frames", type=int, default=1,
                     help="сколько кадров подряд нужно видеть объект, чтобы "
                          "завести трек. 3 отсекает всплеск ложных детекций "
                          "на старте")

    zone = ap.add_argument_group("режим зоны")
    zone.add_argument("--zone", default=None,
                      help="JSON с полигоном. Переключает с линии на зону")
    zone.add_argument("--dwell", type=float, default=2.5,
                      help="сколько секунд в зоне, чтобы засчитать посетителя")
    zone.add_argument("--count-mode", default="dwell-unique",
                      choices=["dwell-unique", "session-max", "hybrid-max"],
                      help="dwell-unique: каждый уникальный трек считается раз "
                           "(теряет счёт при ID switch в плотной группе). "
                           "session-max: за цикл 'зона опустела -> наполнилась -> "
                           "опустела' засчитывается пик одновременных. "
                           "hybrid-max: максимум из двух")
    zone.add_argument("--session-min-seconds", type=float, default=1.0,
                      help="минимальная длительность сессии (отсекает мигания)")

    alert = ap.add_argument_group("тревоги")
    alert.add_argument("--queue-threshold", type=int, default=0,
                       help="тревога, когда в зоне N+ человек одновременно. 0 = выкл")
    alert.add_argument("--queue-duration", type=float, default=30.0,
                       help="сколько секунд очередь должна держаться выше порога")
    alert.add_argument("--absence-zone", default=None,
                       help="JSON с полигоном рабочего места. Тревога, когда "
                            "зона пуста дольше --absence-duration")
    alert.add_argument("--absence-duration", type=float, default=120.0,
                       help="секунд пустоты до тревоги")
    alert.add_argument("--absence-fire-from-start", action="store_true",
                       help="разрешить тревогу до того, как зона хоть раз была "
                            "занята. Нужно для записей, начинающихся с уже "
                            "пустого рабочего места")
    alert.add_argument("--warmup-seconds", type=float, default=0.0,
                       help="глушить все события первые N секунд")

    post = ap.add_argument_group("постобработка лога")
    post.add_argument("--cancel-window", type=int, default=0,
                      help="отмена пар IN<->OUT одного трека внутри N кадров. "
                           "Лечит дрожание рамки у линии. 0 = выкл")
    post.add_argument("--per-track-cooldown", type=int, default=0,
                      help="выбросить повторы в ту же сторону внутри N кадров. "
                           "Лечит застрявший на линии трек. Ориентир 60 (~2 с). "
                           "0 = выкл")
    post.add_argument("--no-dedup", action="store_true",
                      help="отключить дедупликацию по треку — разрешить считать "
                           "одного человека на каждом честном пересечении")

    out = ap.add_argument_group("вывод")
    out.add_argument("--log", default="events.csv", help="CSV с событиями")
    out.add_argument("--show", action="store_true", help="показывать окно")
    out.add_argument("--out-video", default=None, help="записать размеченное видео")
    out.add_argument("--positions-log", default=None,
                     help="CSV с центроидами детекций — сырьё для тепловой карты")
    out.add_argument("--positions-every", type=int, default=15,
                     help="писать позиции раз в N кадров")
    out.add_argument("--max-seconds", type=float, default=0.0,
                     help="остановиться через N секунд видео (0 = без предела)")
    out.add_argument("--preview-line", action="store_true",
                     help="сохранить preview_line.jpg и выйти")

    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.preview_line:
        preview_line(args.source, args.line, args.start, args.end)
        return
    run(args)


if __name__ == "__main__":
    main()

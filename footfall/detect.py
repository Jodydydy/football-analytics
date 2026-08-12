"""
Детекция людей на кадре.

Обёртка над ultralytics YOLO. Тяжёлое (загрузка весов, перенос на GPU)
делается один раз в __init__, лёгкое — на каждом кадре в detect().

Теория: ../footfall-counter/_migration/learn/01-детекция.md
"""
from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector:
    """
    Детектор людей.

        det = Detector("yolo11l.pt", device="0", conf=0.25, imgsz=640)
        detections = det.detect(frame)

    Параметры, которые реально влияют на результат:

    conf    порог уверенности. Выше — меньше мусора, но теряются частично
            загороженные и мелкие люди. Ниже — ловим почти всех, но появляются
            ложные срабатывания. Для подсчёта через линию низкий порог часто
            выгоднее: одиночная ложная детекция не переживёт трекер, а
            пропущенный человек теряется навсегда.

    imgsz   сторона квадрата, во что вписывается кадр (letterbox, с полями).
            При 640 кадр 1920x1080 ужимается втрое: человек высотой 60 px
            становится 20 px и теряется. Отсюда прогоны с imgsz=1920 —
            ценой примерно девятикратного роста вычислений.

    augment TTA: несколько прогонов с преобразованиями. Точнее и заметно
            медленнее.
    """

    def __init__(self,
                 model_path: str = "yolo11l.pt",
                 device: str | None = None,
                 conf: float = 0.25,
                 imgsz: int = 640,
                 classes: list[int] | None = None,
                 augment: bool = False,
                 half: bool = False):
        self.model = YOLO(model_path)

        if device:
            # ultralytics принимает "0", а torch.Module.to() требует "cuda:0"
            self.device = f"cuda:{device}" if device.isdigit() else device
            self.model.to(self.device)
        else:
            self.device = None

        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment
        # half передаём в вызов, а не через model.half(): ultralytics пересобирает
        # и сплавляет модель внутри predict, и на заранее переведённых в fp16
        # весах слияние conv+bn падает
        self.half = half
        # None, а не [0] в сигнатуре: изменяемый объект как значение по умолчанию
        # создаётся один раз при определении функции и становится общим для всех
        # вызовов — классическая ловушка Python
        self.classes = [0] if classes is None else classes

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Найти людей на кадре. Вернуть sv.Detections без tracker_id."""
        # [0] обязателен: модель принимает батч и возвращает список результатов
        results = self.model(frame,
                             classes=self.classes,
                             conf=self.conf,
                             imgsz=self.imgsz,
                             augment=self.augment,
                             half=self.half,
                             verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

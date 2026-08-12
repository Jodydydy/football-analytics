"""
Источники кадров: видеофайл, RTSP-поток, вебкамера, папка с кадрами.

Папка нужна для датасетов вроде MOT17, где последовательность лежит как
img1/000001.jpg, а не как видеофайл. Класс ImageFolderCapture повторяет ту
часть интерфейса cv2.VideoCapture, которой пользуется пайплайн, — поэтому
остальной код не знает, откуда пришёл кадр.
"""
from __future__ import annotations

from pathlib import Path

import cv2


class ImageFolderCapture:
    """Читалка папки с кадрами, совместимая с cv2.VideoCapture.

    Реализует ровно то, что использует пайплайн: isOpened(), read(),
    release(), get(CAP_PROP_*). Кадры сортируются по имени — для MOT17
    и подобных наборов это и есть порядок во времени.
    """

    EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, folder: Path, fps: float = 30.0):
        self.folder = folder
        self.frames = sorted(p for p in folder.iterdir()
                             if p.suffix.lower() in self.EXTS)
        self._idx = 0
        self._fps = fps
        if not self.frames:
            self._w = self._h = 0
            return
        first = cv2.imread(str(self.frames[0]))
        if first is None:
            self._w = self._h = 0
        else:
            self._h, self._w = first.shape[:2]

    def isOpened(self) -> bool:
        return len(self.frames) > 0

    def read(self):
        if self._idx >= len(self.frames):
            return False, None
        frame = cv2.imread(str(self.frames[self._idx]))
        self._idx += 1
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._h
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def release(self):
        pass


def parse_source(s: str):
    """'0' -> 0 (индекс вебкамеры), всё остальное оставить строкой."""
    return int(s) if s.isdigit() else s


def open_source(s):
    """Выбрать читалку по типу источника.

    Папка с картинками -> ImageFolderCapture.
    Всё остальное -> cv2.VideoCapture: файл, RTSP-строка, индекс вебкамеры.
    """
    if isinstance(s, str):
        p = Path(s)
        if p.is_dir() and any(q.suffix.lower() in ImageFolderCapture.EXTS
                              for q in p.iterdir()):
            return ImageFolderCapture(p)
    return cv2.VideoCapture(parse_source(s))

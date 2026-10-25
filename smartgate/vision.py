"""Face detection and embeddings; only load TensorFlow when needed."""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class FaceError(ValueError):
    pass


class FaceEngine:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Не удалось загрузить Haar Cascade из OpenCV.")
        self.model = None
        self.lock = threading.Lock()

    def load_model(self):
        # Only call from the background worker, not from the Tk UI thread.
        with self.lock:
            if self.model is None:
                from keras_facenet import FaceNet
                self.model = FaceNet()

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        rects = self.detector.detectMultiScale(grey, scaleFactor=1.12, minNeighbors=5,
                                                minSize=(55, 55))
        return [tuple(map(int, r)) for r in rects]

    @staticmethod
    def crop_face(bgr: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = rect
        pad = int(max(w, h) * 0.14)
        height, width = bgr.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        result = bgr[y1:y2, x1:x2]
        if result.size == 0:
            raise FaceError("Не удалось выделить область лица.")
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_AREA)

    def embedding(self, rgb_face: np.ndarray) -> np.ndarray:
        self.load_model()
        batch = np.expand_dims(rgb_face, 0)
        with self.lock:
            return np.asarray(self.model.embeddings(batch)[0], dtype=np.float32)

    def enroll_frame(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        faces = self.detect(bgr)
        if len(faces) != 1:
            raise FaceError("На фотографии должно быть ровно одно хорошо видимое лицо. "
                            f"Найдено: {len(faces)}.")
        cropped = self.crop_face(bgr, faces[0])
        return cropped, self.embedding(cropped)

    def enroll_file(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        file = Path(path)
        if not file.is_file() or file.stat().st_size > 12 * 1024 * 1024:
            raise FaceError("Выбери изображение размером до 12 МБ.")
        try:
            with Image.open(file) as img:
                if img.width * img.height > 20_000_000:
                    raise FaceError("Слишком большое изображение (максимум 20 Мп).")
                image = ImageOps.exif_transpose(img).convert("RGB")
                rgb = np.asarray(image)
        except (OSError, ValueError) as exc:
            if isinstance(exc, FaceError):
                raise
            raise FaceError("Не удалось прочитать JPG/PNG фотографию.") from exc
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.enroll_frame(bgr)

    def recognize(self, bgr: np.ndarray,
                  references: list[tuple[int, str, np.ndarray]], threshold: float = 1.0):
        """Recognize visible faces. Return [(rect, user_id | None, name, distance)]."""
        results = []
        for rect in self.detect(bgr)[:5]:
            face_rgb = self.crop_face(bgr, rect)
            vec = self.embedding(face_rgb)
            best_id, best_name, best_dist = None, "Неизвестный", float("inf")
            for user_id, name, ref in references:
                if ref.shape != vec.shape:
                    continue
                dist = float(np.linalg.norm(ref - vec))
                if dist < best_dist:
                    best_id, best_name, best_dist = user_id, name, dist
            if best_dist > threshold:
                best_id, best_name = None, "Неизвестный"
            results.append((rect, best_id, best_name, best_dist))
        return results

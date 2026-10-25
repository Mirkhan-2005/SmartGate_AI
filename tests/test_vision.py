import cv2
import numpy as np
import pytest

from smartgate.vision import FaceEngine, FaceError


def test_no_face_rejected():
    e = FaceEngine()
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    with pytest.raises(FaceError, match="ровно одно"):
        e.enroll_frame(blank)


def test_crop_size():
    src = np.full((250, 300, 3), 128, dtype=np.uint8)
    result = FaceEngine.crop_face(src, (100, 100, 90, 90))
    assert result.shape == (160, 160, 3)


def test_matching_math_without_model():
    e = FaceEngine()
    face = np.zeros((320, 320, 3), dtype=np.uint8)
    e.detect = lambda frame: [(40, 40, 90, 90)]
    e.embedding = lambda crop: np.array([0.5, 0.5, 0.5], dtype=np.float32)
    refs = [(1, "Петя", np.array([0.5, 0.6, 0.5], dtype=np.float32)),
            (2, "Вася", np.array([1.4, 1.2, 1.1], dtype=np.float32))]
    result = e.recognize(face, refs, threshold=0.5)
    assert result[0][1:3] == (1, "Петя")
    assert e.recognize(face, refs, threshold=0.05)[0][2] == "Неизвестный"

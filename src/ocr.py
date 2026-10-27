import re
from typing import Tuple

import cv2
import numpy as np

from src.config import get_settings

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
LATIN_RE = re.compile(r"[A-Za-z]")

_MIN_LONG_SIDE = 1800
_TARGET_LONG_SIDE = 2500
_MAX_UPSCALE = 3.0
_MAX_LONG_SIDE = 4000

_ocr_engine = None


def get_ocr(lang: str = ""):
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        settings = get_settings()
        _ocr_engine = PaddleOCR(
            use_textline_orientation=settings.OCR_USE_TEXTLINE_ORIENTATION,
            lang=lang or settings.OCR_LANG,
        )
    return _ocr_engine


def resize_image(img: np.ndarray) -> np.ndarray:
    long_side = max(img.shape[:2])
    if long_side < _MIN_LONG_SIDE:
        scale = min(_TARGET_LONG_SIDE / long_side, _MAX_UPSCALE)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if long_side > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / long_side
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _group_lines(texts, boxes):
    if not boxes or len(boxes) != len(texts):
        return [" ".join(texts)]
    items = []
    for t, b in zip(texts, boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        items.append((t, (y1 + y2) / 2.0, (x1 + x2) / 2.0, max(y2 - y1, 1.0)))
    items.sort(key=lambda i: i[1])
    heights = sorted(i[3] for i in items)
    threshold = heights[len(heights) // 2] * 0.5

    rows = [[items[0]]]
    for it in items[1:]:
        if abs(it[1] - rows[-1][0][1]) < threshold:
            rows[-1].append(it)
        else:
            rows.append([it])

    lines = []
    for row in rows:
        joined = " ".join(i[0] for i in row)
        rtl = len(ARABIC_RE.findall(joined)) > len(LATIN_RE.findall(joined))
        row.sort(key=lambda i: i[2], reverse=rtl)
        lines.append(" ".join(i[0] for i in row))
    return lines


def read_image(img: np.ndarray, lang: str = "") -> Tuple[str, float]:
    ocr = get_ocr(lang)
    texts, scores, boxes = [], [], []
    for res in ocr.predict(resize_image(img)):
        texts.extend(res["rec_texts"])
        scores.extend(float(s) for s in res["rec_scores"])
        boxes.extend(res["rec_boxes"])

    weight = sum(len(t) for t in texts)
    confidence = (
        sum(s * len(t) for s, t in zip(scores, texts)) / weight if weight else 0.0
    )
    return "\n".join(_group_lines(texts, boxes)), round(confidence, 4)

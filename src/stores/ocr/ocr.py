import re
from typing import Tuple

import cv2
import numpy as np

from config import get_settings

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
LATIN_RE = re.compile(r"[A-Za-z]")

_MIN_LONG_SIDE = 1200
_TARGET_LONG_SIDE = 1600
_MAX_UPSCALE = 2.0
_MAX_LONG_SIDE = 2000

_ocr_engines = {}


def get_ocr(lang: str = ""):
    global _ocr_engines
    settings = get_settings()
    resolved_lang = lang or settings.OCR_LANG
    if resolved_lang not in _ocr_engines:
        from paddleocr import PaddleOCR

        kwargs = {
            "lang": resolved_lang,
            "use_doc_orientation_classify": settings.OCR_USE_DOC_ORIENTATION,
            "use_doc_unwarping": settings.OCR_USE_DOC_UNWARPING,
            "use_textline_orientation": settings.OCR_USE_TEXTLINE_ORIENTATION,
        }
        if settings.OCR_DET_MODEL:
            kwargs["text_detection_model_name"] = settings.OCR_DET_MODEL
        if settings.OCR_DET_LIMIT_SIDE_LEN and settings.OCR_DET_LIMIT_SIDE_LEN > 0:
            kwargs["text_det_limit_side_len"] = settings.OCR_DET_LIMIT_SIDE_LEN
        _ocr_engines[resolved_lang] = PaddleOCR(**kwargs)
    return _ocr_engines[resolved_lang]


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

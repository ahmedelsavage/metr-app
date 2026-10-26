from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ocr import _sort_reading_order, extract_text
from schemas import OCRResult


def _make_mock_ocr_result(texts, scores, boxes):
    res = MagicMock()
    res.__getitem__ = lambda self, key: {
        "rec_texts": texts,
        "rec_scores": scores,
        "rec_boxes": boxes,
    }[key]
    return res


def test_reading_order_arabic_rtl():
    texts = ["يسار", "يمين"]
    scores = [0.95, 0.90]
    boxes = [[10, 10, 100, 40], [200, 10, 300, 40]]
    text, score = _sort_reading_order(texts, scores, boxes)
    assert text == "يمين يسار"
    assert score == 0.925


def test_reading_order_english_ltr():
    texts = ["world", "hello"]
    scores = [0.90, 0.90]
    boxes = [[200, 10, 300, 40], [10, 10, 100, 40]]
    text, score = _sort_reading_order(texts, scores, boxes)
    assert text == "hello world"


def test_reading_order_vertical():
    texts = ["second line", "first line"]
    scores = [0.8, 0.9]
    boxes = [[10, 50, 100, 80], [10, 10, 100, 40]]
    text, score = _sort_reading_order(texts, scores, boxes)
    assert text == "first line\nsecond line"


def test_missing_image_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "non_existent.png")


@patch("ocr.get_ocr")
def test_extract_text_mocked(mock_get_ocr, tmp_path):
    test_img = tmp_path / "sample.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    mock_engine = MagicMock()
    mock_engine.predict.return_value = [
        _make_mock_ocr_result(
            texts=["عقد بيع", "Sale Contract"],
            scores=[0.95, 0.92],
            boxes=[[10, 10, 200, 40], [10, 50, 200, 80]],
        )
    ]
    mock_get_ocr.return_value = mock_engine

    result = extract_text(test_img)
    assert isinstance(result, OCRResult)
    assert "عقد بيع" in result.text
    assert "Sale Contract" in result.text
    assert result.confidence == 0.935
    assert result.language == "ar"

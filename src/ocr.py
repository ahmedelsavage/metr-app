import re
from pathlib import Path

from src.config import get_settings
from src.schemas import OCRResult

_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_ocr_engine = None


def get_ocr(lang: str):
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        settings = get_settings()
        _ocr_engine = PaddleOCR(
            use_textline_orientation=settings.OCR_USE_TEXTLINE_ORIENTATION,
            lang=lang or settings.OCR_LANG,
        )
    return _ocr_engine


def detect_language(text: str) -> str:
    if _ARABIC_PATTERN.search(text):
        return "ar"
    else:
        return "en"


def _sort_reading_order(
    texts: list[str], scores: list[float], boxes: list
) -> tuple[str, float]:
    if not texts:
        return "", 1.0
    if not boxes or len(boxes) != len(texts):
        return "\n".join(texts), round(sum(scores) / len(scores), 4)

    items = []
    for text, score, box in zip(texts, scores, boxes):
        coords = [float(b) for b in box]
        y_center = (coords[1] + coords[3]) / 2.0
        x_center = (coords[0] + coords[2]) / 2.0
        height = max(coords[3] - coords[1], 1.0)
        items.append((text, float(score), y_center, x_center, height))

    items.sort(key=lambda x: x[2])
    row_threshold = (sum(i[4] for i in items) / len(items)) * 0.5

    rows = [[items[0]]]
    for item in items[1:]:
        if abs(item[2] - rows[-1][-1][2]) < row_threshold:
            rows[-1].append(item)
        else:
            rows.append([item])

    lines = []
    for row in rows:
        row_text = " ".join(item[0] for item in row)
        is_arabic = bool(_ARABIC_PATTERN.search(row_text))
        row.sort(key=lambda x: x[3], reverse=is_arabic)
        lines.append(" ".join(item[0] for item in row))

    avg_score = sum(i[1] for i in items) / len(items)
    return "\n".join(lines), round(avg_score, 4)


def extract_text(image: str, lang="") -> OCRResult:
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_input = str(img_path)
    else:
        image_input = image

    ocr = get_ocr(lang=lang)
    raw_results = ocr.predict(image_input)

    texts, scores, boxes = [], [], []
    for res in raw_results:
        if hasattr(res, "__getitem__"):
            texts.extend(res["rec_texts"])
            scores.extend(res["rec_scores"])
            if "rec_boxes" in res:
                boxes.extend(res["rec_boxes"])

    full_text, confidence = _sort_reading_order(texts, scores, boxes)
    return OCRResult(
        text=full_text,
        confidence=confidence,
        language=detect_language(full_text),
    )


def save_outputs(result, source_path, out_dir="reports/ocr"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(source_path).stem
    text_file = out_dir / f"{base_name}_ocr.txt"
    markdown_file = out_dir / f"{base_name}_ocr.md"

    with open(text_file, "w", encoding="utf-8") as f:
        f.write(result.text)
        f.write(f"\n\nConfidence: {result.confidence}\nLanguage: {result.language}\n")

    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(f"# OCR Results for {base_name}\n\n")
        f.write(f"**Confidence:** {result.confidence}\n\n")
        f.write(f"**Language:** {result.language}\n\n")
        f.write("## Extracted Text\n\n")
        f.write(result.text)

    print(f"OCR results saved to: {text_file} and {markdown_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = extract_text(sys.argv[1])
        save_outputs(res, sys.argv[1])
    else:
        print("Usage: python -m src.ocr <image_path>")

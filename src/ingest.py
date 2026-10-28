from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pymupdf

from stores.ocr import ARABIC_RE, LATIN_RE, read_image
from schemas import OCRResult

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_PDF_DPI = 300
_MAX_PAGES = 20
_MAX_FILE_MB = 15


def detect_language(text: str) -> str:
    ar = len(ARABIC_RE.findall(text))
    la = len(LATIN_RE.findall(text))
    total = ar + la
    if total < 10:
        return "unknown"
    if ar / total > 0.7:
        return "ar"
    if la / total > 0.7:
        return "en"
    return "mixed"


def detect_file_type(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(5)
    if header == b"%PDF-":
        return "pdf"
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix == ".txt":
        return "text"
    return "unknown"


def _read_pdf(path: Path, lang: str):
    texts, scores = [], []
    with pymupdf.open(str(path)) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        if len(doc) > _MAX_PAGES:
            raise ValueError(f"PDF has {len(doc)} pages (max {_MAX_PAGES})")
        for page in doc:
            pix = page.get_pixmap(dpi=_PDF_DPI, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            text, conf = read_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), lang)
            texts.append(text)
            scores.append(conf)
    return "\n\n".join(texts), sum(scores) / len(scores) if scores else 0.0


def process_file(path: Union[str, Path], lang: str = "") -> OCRResult:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise ValueError("File is empty")
    if size > _MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File too large (max {_MAX_FILE_MB} MB)")

    kind = detect_file_type(p)
    if kind == "pdf":
        text, conf = _read_pdf(p, lang)
    elif kind == "image":
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Cannot read image: {p}")
        text, conf = read_image(img, lang)
    elif kind == "text":
        text, conf = p.read_text(encoding="utf-8-sig", errors="replace"), 1.0
    else:
        raise ValueError(f"Unsupported file type: {p.suffix or 'unknown'}")

    return OCRResult(
        text=text, confidence=round(conf, 4), language=detect_language(text)
    )


def save_outputs(result: OCRResult, source_path, out_dir="reports/ocr"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(source_path).stem
    (out / f"{name}_ocr.txt").write_text(
        f"{result.text}\n\nConfidence: {result.confidence}\nLanguage: {result.language}\n",
        encoding="utf-8",
    )
    print(f"Saved to {out / (name + '_ocr.txt')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = process_file(sys.argv[1])
        save_outputs(res, sys.argv[1])
    else:
        print("Usage: python -m src.ingest <file_path>")

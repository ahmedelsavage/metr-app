import re
from typing import List

DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

ARTICLE_SPLIT = re.compile(r"(?=^\s*ماد[ةه]\s*\(\s*[0-9٠-٩۰-۹]+\s*\))", re.MULTILINE)
ARTICLE_HEAD = re.compile(r"^\s*ماد[ةه]\s*\(\s*([0-9٠-٩۰-۹]+)\s*\)")

MIN_LEN = 15


def by_article(text: str, size: int, overlap: int) -> List[dict]:
    chunks = []
    for part in ARTICLE_SPLIT.split(text):
        part = part.strip()
        if len(part) < MIN_LEN:
            continue
        m = ARTICLE_HEAD.match(part)
        chunks.append({
            "text": part,
            "article_number": m.group(1).translate(DIGITS) if m else None,
        })
    return chunks


def by_chars(text: str, size: int, overlap: int) -> List[dict]:
    step = max(size - overlap, 1)
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if len(piece) >= MIN_LEN:
            chunks.append({"text": piece, "article_number": None})
    return chunks


def by_words(text: str, size: int, overlap: int) -> List[dict]:
    words = text.split()
    step = max(size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size]).strip()
        if len(piece) >= MIN_LEN:
            chunks.append({"text": piece, "article_number": None})
    return chunks


STRATEGIES = {
    "article": by_article,
    "fixed_chars": by_chars,
    "words": by_words,
}


def chunk_text(text: str, strategy: str, size: int, overlap: int) -> List[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, size, overlap)

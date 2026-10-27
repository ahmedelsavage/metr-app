import re
from typing import List

DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

ARTICLE_SPLIT = re.compile(r"(?=\[[^\]\n]+\]\s*ماد[ةه]\s*\(\s*[0-9٠-٩۰-۹]+\s*\))")
ARTICLE_HEAD = re.compile(r"^\[([^\]\n]+)\]\s*ماد[ةه]\s*\(\s*([0-9٠-٩۰-۹]+)\s*\)")
HEAD_ANY = re.compile(r"\[([^\]\n]+)\]\s*ماد[ةه]\s*\(\s*([0-9٠-٩۰-۹]+)\s*\)")

MIN_LEN = 15


def _index(text: str):
    return [(m.start(), m.group(1).strip(), m.group(2).translate(DIGITS)) for m in HEAD_ANY.finditer(text)]


def _owner(index, pos: int):
    law, art = None, None
    for start, l, a in index:
        if start <= pos:
            law, art = l, a
        else:
            break
    return law, art


def by_article(text: str, size: int, overlap: int) -> List[dict]:
    chunks = []
    for part in ARTICLE_SPLIT.split(text):
        part = part.strip()
        if len(part) < MIN_LEN:
            continue
        m = ARTICLE_HEAD.match(part)
        if not m:
            continue
        chunks.append({
            "text": part,
            "law_name": m.group(1).strip(),
            "article_number": m.group(2).translate(DIGITS),
        })
    return chunks


def by_chars(text: str, size: int, overlap: int) -> List[dict]:
    index = _index(text)
    step = max(size - overlap, 1)
    chunks = []
    for i in range(0, len(text), step):
        piece = text[i:i + size].strip()
        if len(piece) < MIN_LEN:
            continue
        law, art = _owner(index, i)
        chunks.append({"text": piece, "law_name": law, "article_number": art})
    return chunks


def by_words(text: str, size: int, overlap: int) -> List[dict]:
    index = _index(text)
    words = text.split()
    step = max(size - overlap, 1)
    chunks = []
    pos = 0
    offsets = []
    for w in words:
        pos = text.find(w, pos)
        offsets.append(pos)
        pos += len(w)
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + size]).strip()
        if len(piece) < MIN_LEN:
            continue
        law, art = _owner(index, offsets[i])
        chunks.append({"text": piece, "law_name": law, "article_number": art})
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

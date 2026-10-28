import sys
from pathlib import Path

import pymupdf

source = sys.argv[1]

with pymupdf.open(source) as doc:
    text = "\n".join(page.get_text() for page in doc)

Path("pdf_text_debug.txt").write_text(text, encoding="utf-8")
print("حروف النص:", len(text))
print(text[:600])

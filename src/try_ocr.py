import sys
from pathlib import Path

from ingest import process_file
from pipeline import split_clauses

source = sys.argv[1]
result = process_file(source)

Path("ocr_debug.txt").write_text(result.text, encoding="utf-8")

clauses = split_clauses(result.text)

print("confidence:", result.confidence)
print("language:", result.language)
print("حروف النص:", len(result.text))
print("عدد البنود:", len(clauses))

for i, clause in enumerate(clauses, 1):
    print(i, len(clause), clause[:60])

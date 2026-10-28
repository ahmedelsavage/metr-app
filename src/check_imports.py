import os

from config import get_settings
from stores.llm import build_risk_prompt
from stores.ocr import read_image
from stores.rag import RAG

import ingest
import pipeline

print("config:", get_settings().VECTOR_DB_PATH)
print("pipeline:", pipeline.split_clauses.__name__)
print("ocr:", read_image.__name__, "| ingest:", ingest.__name__)
print("env:", bool(os.getenv("QWEN8_API_URL")), bool(os.getenv("QWEN8_API_KEY")))
print("prompt:", len(build_risk_prompt("عقد تجريبي", "سياق تجريبي", "ar")), "حرف")

rag = RAG()
results = rag.search("يجوز الاتفاق على أن يعتبر العقد مفسوخا من تلقاء نفسه", limit=2)
print("rag:", [(r.law_name, r.article_number) for r in results])

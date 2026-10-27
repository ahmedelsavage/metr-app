import json
import sys
import time
from pathlib import Path

from llm import generate_response
from pipeline import analyze_contract_text
from rag import RAG

LIMIT_PER_CLAUSE = 2
FILE_NAME = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.txt"

raw_outputs = []


def recording_generate(prompt: str) -> str:
    print("حجم الـ prompt:", len(prompt), "حرف")
    started = time.perf_counter()
    output = generate_response(prompt)
    print(f"رد من السيرفر بعد {time.perf_counter() - started:.0f} ثانية | {len(output)} حرف")
    raw_outputs.append(output)
    return output


try:
    print("اختبار الاتصال:", generate_response("قل مرحبا في كلمة واحدة")[:200])
except Exception as error:
    print("فشل الاتصال:", error)
    raise SystemExit(1)

rag = RAG()
text = Path(FILE_NAME).read_text(encoding="utf-8")
print("الملف:", FILE_NAME)

try:
    result = analyze_contract_text(
        text,
        rag,
        limit_per_clause=LIMIT_PER_CLAUSE,
        generate_fn=recording_generate,
    )
except ValueError as error:
    Path("raw_outputs.txt").write_text("\n\n=====\n\n".join(raw_outputs), encoding="utf-8")
    print("فشل التحليل:", error)
    raise SystemExit(1)

Path("raw_outputs.txt").write_text("\n\n=====\n\n".join(raw_outputs), encoding="utf-8")
Path("result_debug.json").write_text(
    json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
)

print()
print("contract_type:", result.contract_type)
print("verdict:", result.verdict)
print("summary:", result.summary)
print()

for i, risk in enumerate(result.risks, 1):
    print(f"{i}. {risk.risk_type} | {risk.severity} | {risk.legal_reference}")
    print("   البند:", risk.clause)
    print("   السبب:", risk.reason)

print()
print("advice:", result.advice)

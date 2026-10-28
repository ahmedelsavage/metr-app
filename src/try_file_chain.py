import json

from pipeline import analyze_file
from stores.rag import RAG

captured = {}

FAKE_RESULT = {
    "contract_type": "rent",
    "summary": "ملخص تجريبي",
    "risks": [
        {
            "clause": "بند تجريبي",
            "reason": "سبب تجريبي",
            "risk_type": "unilateral_termination",
            "severity": "high",
            "legal_reference": None,
        }
    ],
    "verdict": "risky",
    "advice": "نصيحة تجريبية",
}


def fake_generate(prompt: str) -> str:
    captured["prompt"] = prompt
    return json.dumps(FAKE_RESULT, ensure_ascii=False)


rag = RAG()
result = analyze_file(
    "rag_test_samples/01_sale_risky.txt", rag, limit_per_clause=2, generate_fn=fake_generate
)

prompt = captured["prompt"]
print("verdict:", result.verdict, "| عدد المخاطر:", len(result.risks))
print("طول الـ prompt:", len(prompt), "حرف")
print("عدد المواد في السياق:", prompt.count("] مادة ("))

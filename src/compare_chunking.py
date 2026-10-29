import json
import sys
import time
from pathlib import Path

from pipeline import build_legal_context, split_clauses
from stores.llm import analyze_contract
from stores.rag import RAG

BASE = Path(__file__).resolve().parent
KB_DIR = BASE.parent / "data" / "laws_raw"
EVAL_DIR = BASE / "rag_eval"
OUT_FILE = BASE / "chunking_compare.json"

STRATEGIES = ["article", "fixed_chars", "words"]
WORDS_SIZE = 150
WORDS_OVERLAP = 15
RETRIEVE_K = 5
LLM_LIMIT = 1


def ensure_collections(rag):
    for strategy in STRATEGIES:
        if rag.client.collection_exists(rag.collection(strategy)):
            continue
        size, overlap = rag.s.CHUNK_SIZE, rag.s.CHUNK_OVERLAP
        if strategy == "words":
            rag.s.CHUNK_SIZE, rag.s.CHUNK_OVERLAP = WORDS_SIZE, WORDS_OVERLAP
        count = rag.build_kb(str(KB_DIR), strategy)
        rag.s.CHUNK_SIZE, rag.s.CHUNK_OVERLAP = size, overlap
        print(f"built {strategy}: {count} chunks")


def first_hit_rank(chunks, expected_articles):
    wanted = {(a["law"], str(a["article"])) for a in expected_articles}
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk.law_name, str(chunk.article_number)) in wanted:
            return rank
    return None


def run_one(rag, strategy, text, item):
    clauses = split_clauses(text)
    per_clause = [rag.search(c, limit=RETRIEVE_K, strategy=strategy) for c in clauses]

    ranks = []
    if len(clauses) == len(item["clauses"]):
        for found, spec in zip(per_clause, item["clauses"]):
            if spec["risk_type"]:
                ranks.append(first_hit_rank(found, spec["expected_articles"]))

    context_chunks = [chunk for found in per_clause for chunk in found[:LLM_LIMIT]]
    legal_context = build_legal_context(context_chunks)

    row = {
        "clauses_found": len(clauses),
        "ranks": ranks,
        "context_chars": len(legal_context),
        "predicted_risks": [],
        "verdict": None,
        "error": None,
    }

    start = time.perf_counter()
    try:
        analysis = analyze_contract(
            contract_text=text, legal_context=legal_context, language="ar"
        )
        row["predicted_risks"] = sorted({r.risk_type for r in analysis.risks})
        row["verdict"] = analysis.verdict
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:300]
    row["latency"] = round(time.perf_counter() - start, 1)
    return row


def hit_rate(ranks, k):
    if not ranks:
        return 0.0
    return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3)


def summarize(runs, expected):
    summary = {}
    for strategy in STRATEGIES:
        rows = [r for r in runs.values() if r["strategy"] == strategy]
        ok = [r for r in rows if not r["error"]]
        ranks = [rank for r in rows for rank in r["ranks"]]

        tp = fp = fn = correct = 0
        for r in ok:
            exp = expected[r["id"]]
            want = set(exp["expected_risks"])
            got = set(r["predicted_risks"])
            tp += len(want & got)
            fp += len(got - want)
            fn += len(want - got)
            correct += r["verdict"] == exp["expected_verdict"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mrr = sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0

        summary[strategy] = {
            "contracts": len(rows),
            "llm_errors": len(rows) - len(ok),
            "risky_clauses": len(ranks),
            "hit@1": hit_rate(ranks, 1),
            "hit@3": hit_rate(ranks, 3),
            "hit@5": hit_rate(ranks, 5),
            "mrr": round(mrr, 3),
            "verdict_accuracy": round(correct / len(ok), 3) if ok else 0.0,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "avg_context_chars": round(sum(r["context_chars"] for r in rows) / len(rows)) if rows else 0,
            "avg_latency": round(sum(r["latency"] for r in rows) / len(rows), 1) if rows else 0.0,
        }
    return summary


def save(runs, expected):
    payload = {"summary": summarize(runs, expected), "runs": runs}
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    items = json.loads((EVAL_DIR / "expected.json").read_text(encoding="utf-8"))
    expected = {item["id"]: item for item in items}
    selected = items[:limit] if limit else items

    runs = {}
    if OUT_FILE.exists():
        runs = json.loads(OUT_FILE.read_text(encoding="utf-8")).get("runs", {})

    rag = RAG()
    ensure_collections(rag)

    for item in selected:
        text = (EVAL_DIR / item["file"]).read_text(encoding="utf-8")
        for strategy in STRATEGIES:
            key = f"{item['id']}|{strategy}"
            if key in runs and not runs[key]["error"]:
                continue
            row = run_one(rag, strategy, text, item)
            row["id"] = item["id"]
            row["strategy"] = strategy
            runs[key] = row
            print(key, row["error"] or row["verdict"], f"{row['latency']}s")
            save(runs, expected)

    save(runs, expected)
    print(json.dumps(summarize(runs, expected), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from typing import List, Optional
import re
from schemas import RetrievedChunk, ContractAnalysis
from stores.llm import analyze_contract
from stores.rag import RAG
from ingest import process_file

def deduplicate_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    best = {}

    for chunk in chunks:
        key = (chunk.law_name, chunk.article_number or chunk.text)

        if key not in best or chunk.score > best[key].score:
            best[key] = chunk

    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def build_legal_context(chunks: List[RetrievedChunk], max_chunks: Optional[int] = None) -> str:
    unique = deduplicate_chunks(chunks)

    if max_chunks:
        unique = unique[:max_chunks]

    return "\n\n".join(chunk.text for chunk in unique)

CLAUSE_START = re.compile(r"(?m)^[^\S\n]*[^\s:]{0,3}بند\s+[^:\n]{1,30}:")

def split_clauses(text: str, min_length: int = 40) -> List[str]:
    starts = [m.start() for m in CLAUSE_START.finditer(text)]

    if starts:
        bounds = starts + [len(text)]
        parts = [text[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
    else:
        parts = re.split(r"\n\s*\n", text)

    return [" ".join(part.split()) for part in parts if len(part.strip()) >= min_length]

def gather_legal_chunks(
    rag: RAG, clauses: List[str], limit_per_clause: int = 5
) -> List[RetrievedChunk]:
    chunks = []

    for clause in clauses:
        chunks.extend(rag.search(clause, limit=limit_per_clause))

    return chunks


def analyze_contract_text(
    contract_text: str,
    rag: RAG,
    language: str = "ar",
    limit_per_clause: int = 5,
    generate_fn=None,
) -> ContractAnalysis:
    clauses = split_clauses(contract_text)
    chunks = gather_legal_chunks(rag, clauses, limit_per_clause)
    legal_context = build_legal_context(chunks)

    return analyze_contract(
        contract_text=contract_text,
        legal_context=legal_context,
        language=language,
        generate_fn=generate_fn,
    )

def analyze_file(
    path,
    rag: RAG,
    language: str = "ar",
    limit_per_clause: int = 5,
    generate_fn=None,
) -> ContractAnalysis:
    ocr_result = process_file(path)

    return analyze_contract_text(
        ocr_result.text,
        rag,
        language=language,
        limit_per_clause=limit_per_clause,
        generate_fn=generate_fn,
    )
